#!/usr/bin/env python
import json
import random
from util.util import safe_mkdir, removekey
from pprint import pprint


def remove_ignored(signs):
    new = {}
    n_entries = 0
    n_signs = 0

    for image in sorted(signs):
        #accumulate all non-ignored signs in current frame 
        signs_list = [sign for sign in signs[image] if sign['ignore'] == False]
        n_signs += len(signs_list)

        #add non-ignored sign list for current frame (if not empty)
        if len(signs_list) > 0:
            n_entries += 1
            new[image] = signs_list


    print("Pictures without ignored = {}".format(n_entries))
    print("Number of signs without ignored = {}\n".format(n_signs))
    # pprint(new)
    return new


def load_marking(filename):
    with open(filename) as f:
        signs = json.load(f)
    print("Marking is loaded. ")
    return signs



# def read_classes(path):
#     classes = set()
#     with open(path) as f:
#         lines = f.readlines()[1:]
#         for line in lines:
#             s = line.split(",")
#             classes.add(s[1].strip())

#     return classes



# def get_label_list(marking):
#     labels = set()
#     last = set()
#     for name in sorted(marking):
#         for sign_entry in marking[name]:
#             class_name = sign_entry["sign_class"]
#             if "unknown" in class_name:
#                 last.add(class_name)
#             else:
#                 labels.add(class_name)
#     # print(labels)         
#     return sorted(labels) + sorted(last)



# def get_label_set(marking):
#     return set(get_label_list(marking))




# def DictForClasses(classes):
#     d = {}
#     for name in sorted(classes):
#         if not "unknown" in name:
#             d[name]  = []
#     return d



# def save_stats(norm,  rootpath):
#     with open("{}/stats.txt".format(rootpath), 'w') as f:
#         stats = {}
#         for label in sorted(norm["test"]):
#             train_size = len(norm["train"][label])
#             test_size = len(norm["test"][label])
#             total = test_size + train_size
#             stats[label] = "{:10}{:8}{:7}{:7}{:14.2}".format(label, train_size, test_size, total, float(test_size)/total)

#         first = "{:10}{:>8}{:>7}{:>7}{:>14}". format("label", "train", "test", "total", "test/total")
#         f.write(first+'\n')
#         print(first)
#         for label in sorted(stats):
#             f.write(stats[label] + '\n')
#             print(stats[label])




# #marking = dict of 2 dicts (train and test dicts each containing lists of sign entries)
# #swaps train/test for classes in which train < test; removes classes with toofew images
# def  normalize(markings, classes):
#     threshold = 10
#     norm = {}; norm["train"], norm["test"] = {},{}

#     for label in classes:
#         train_size = len(markings["train"][label])
#         test_size = len(markings["test"][label])

#         #cut off classes with few images
#         if (train_size + test_size) > threshold:
#             if train_size < test_size: #then swap
#                 norm["train"][label], norm["test"][label] = markings["test"][label], markings["train"][label]
#                 train_size, test_size = test_size, train_size
#             else:
#                 norm["train"][label], norm["test"][label] = markings["train"][label], markings["test"][label]

#     save_stats(norm, rootpath)
#     return norm




# #here marking is dict of lists of tuples: label: [(filename, entry_for_marking)]
# def to_std_marking(marking):
#     filenames = set()
#     for label in sorted(marking): # for each sign class
#         for entry in marking[label]: #for each class member
#             filenames.add(entry[0]) #adding entry for filename


#     new = {}
#     i = 0
#     for name in sorted(filenames):
#         new[name] = []
#     for label in sorted(marking):
#         for entry in marking[label]:
#             new[entry[0]] += [entry[1]]
#             i += 1


#     print("number of images in set = ", i)
#     return new





# def classification_marking(marking, classes):
#     # new = marking
#     new = {}
#     for phase in ['train', 'test']:
#         cdict = DictForClasses(classes)
#         m = marking[phase]
#         for name in sorted(m):
#             for sign_entry in m[name]:
#                 label =  sign_entry['sign_class']
#                 if not ("unknown" in label) and (label in classes):
#                     cdict[label] += [(name, sign_entry)]
#         new[phase] = cdict

#     new = normalize(new, classes)
#     new['train'] = to_std_marking(new['train'])
#     new['test'] = to_std_marking(new['test'])

#     return new


         
def get_classification_labels(classes):
    common = classes['train'].intersection(classes['test'])
    tmp = set()
    for label in common:
        if not "unknown" in label:
            tmp.add(label)

    return tmp



def gather_signs(path):
    res = {}
    for phase in ["train", "test"]:
        filename = "{}/marking_{}.json".format(path, phase)
        m = remove_ignored(load_marking(filename))
        for pict_name in sorted(m):
            for sign in m[pict_name]:
                sign['pict_name'] = pict_name
                #strip: "unknown_unmarked" should be equal "unknown_unmarked " (occures in original marking)
                res.setdefault(sign["sign_class"].strip(), []).append(sign) 
    return res


def remove_unknown(classes):
    for class_name in sorted(classes):
        if "unknown" in class_name:
            classes = removekey(classes, class_name)

    print("Number of classes after removing unknown: {}".format(len(sorted(classes))))
    return classes
           

def remove_small(classes, threshold):
    for class_name in sorted(classes):
        if len(classes[class_name]) < threshold:
            classes = removekey(classes, class_name)

    msg = "Number of classes after removing small: {}\nThreshold = {}".format(len(sorted(classes)), threshold)
    print(msg)
    return classes


def sort_by_id(unsorted):
    res = {}
    for sign in unsorted:
        res.setdefault(str(sign["sign_id"]), []).append(sign) 
    return res

from bisect import bisect


def get_random_suitable(cl, size):
    #sort phys signs according to their nomber of instances]
    sorted_keys = [i[0] for i in sorted(cl.items(), key=lambda x: len(x[1]))]
    lengths = [len(cl[i]) for i in sorted_keys] 

    # find the position where suitable sign can be found. 
    # if all signs have more instances than size then pos = 0(take minimum of sign instances)
    pos = bisect(lengths, size)

    # if pos == i then we should consider all elements from 0 to i-1: [0:i]. 
    # if pos == 0 then there is no suitable elements => we should consider only [0:1] ([0:0] makes empty list)
    high_ind = max(1, pos) 
    phys_sign = random.choice(sorted_keys[0:high_ind+1])

    return phys_sign
    

def unsort_idsorted(idsorted):
    res = []
    for phys_sign in sorted(idsorted):
        res += idsorted[phys_sign]
    return res


def split_class(total, ratio):
    total_size = len(total)
    part_size = int(total_size * ratio) 
    curr_size = 0
    rest = sort_by_id(total)
    part = {}

    while curr_size < part_size:
        phys_sign = get_random_suitable(rest, part_size - curr_size)
        part[phys_sign] = rest[phys_sign]
        curr_size += len(rest[phys_sign])
        del rest[phys_sign]
    part = unsort_idsorted(part) 
    rest = unsort_idsorted(rest)

    return part, rest


def split_proportionally(total, ratio):
    total_part, total_rest = {}, {}

    for class_name in sorted(total):
        class_part, class_rest = split_class(total[class_name], ratio)
        # print(len(total[class_name]), len(class_part), len(class_rest))
        total_part[class_name] = class_part
        total_rest[class_name] = class_rest

    return total_part, total_rest
    

def split(classes, test_ratio=0.2, val_ratio=0):
    print("splitting: test/rest")
    test, rest = split_proportionally(classes, test_ratio)
    print("splitting: val/train")
    val, train = split_proportionally(rest, val_ratio)

    return train, test, val
        
  


def classification_marking(path, threshold=100, seed=42, test_ratio=0.2, val_ratio=0.1):
    random.seed(seed)
    gathered = gather_signs(path) #gather signs from 2 markings into one dictionary (key = sign_class)
    print("Number of classes : {}".format(len(sorted(gathered))))
    reduced = remove_small(remove_unknown(gathered), threshold)

    train, test, val = split(reduced, test_ratio=test_ratio, val_ratio=val_ratio)
    marking = {'train': train, 'val': val, 'test': test}
    return marking



def save_marking(marking, path, prefix):
    print('saving markings')
    for phase in sorted(marking):
        filename = "{}/{}_{}.json".format(path, prefix, phase)
        with open(filename, 'w') as f:
            content = json.dumps(marking[phase], indent=2, sort_keys=True)
            f.write(content)



if __name__ == '__main__':
    print("detection marking -> classification marking")
    rootpath = '../global_data/Traffic_signs/RTSD'
    marking = classification_marking(rootpath, threshold=100, seed=42)
    save_marking(marking, rootpath)
    