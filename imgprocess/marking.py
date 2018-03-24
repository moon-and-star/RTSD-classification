# !/usr/bin/env python
import json
import random
from util.util import safe_mkdir, removekey
from pprint import pprint
import os.path as osp


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
    
    return new


def load_marking(filename):
    with open(filename) as f:
        signs = json.load(f)
    print("Marking is loaded.")
    return signs
    

def load_classification_marking(marking_path, prefix):
    marking = {}

    for phase in ['train', 'test', 'val']:
        path = '{}/{}_{}.json'.format(marking_path, prefix, phase)
        if osp.exists(path):
            marking[phase] = load_marking(path)
        else:
            marking[phase] = None
            print("WARNING: no {} marking file found.".format(phase))
    
    return marking

         
def get_classification_labels(classes):
    common = classes['train'].intersection(classes['test'])
    tmp = set()
    for label in common:
        if not "unknown" in label:
            tmp.add(label)

    return tmp


# this function is for gathering marking from 2 marking files: for train and for test
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
           
#this function had been used when there was only 1 threshold for class size:
# total number of sign images in class
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

# old function for unified splitting for all classes
# usually used after removing small classes
#propotional split
def val_split_class(total, ratio):
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

    # if all physical signs fell into validation part (rest is empty)
    # this means that this class is too small for that ratio
    if len(rest) == 0:
        rest, part = part, rest

    part = unsort_idsorted(part)
    rest = unsort_idsorted(rest)

    return part, rest

def val_split(total, ratio):
    total_part, total_rest = {}, {}

    for class_name in sorted(total):
        class_part, class_rest = val_split_class(total[class_name], ratio)
        # print(len(total[class_name]), len(class_part), len(class_rest))
        total_part[class_name] = class_part
        total_rest[class_name] = class_rest

    return total_rest, total_part


# extracts max represented sign from class and returns it
def take_out_max(idsorted):
    max = 0
    id_of_max = sorted(idsorted)[0]

    for id in idsorted:
        if max < len(idsorted[id]): # if found physical sign with n=more images
            max = len(idsorted[id])
            id_of_max = id

    sign_imgs = idsorted[id_of_max]
    del idsorted[id_of_max]

    return id_of_max, sign_imgs

# takes out set of physical signs that satisfies specified thresholds
#uses greedy algorithm
def greedy_selection(idsorted, min_phys=5., min_imgs=25.):
    part = {}
    curr_imgs = 0
    curr_phys = 0
    while curr_imgs < min_imgs or curr_phys < min_phys:  # greedy selection of physical sign
        sign_id, images = take_out_max(idsorted)
        part[sign_id] = images
        curr_phys += 1
        curr_imgs += len(images)

    return part



# new version of class splitter: multiple thresholds
# (min/max for both physical signs and sign images inside class)
# TODO add threshold parameters
def tsplit_class(total):
    idsorted = sort_by_id(total)
    num_of_imgs = len(total)
    num_of_phys = len(set(sorted(idsorted)))
    rest = {}

    if num_of_phys < 5 or num_of_imgs < 25:
        part = idsorted
    elif num_of_imgs < 125:
        part = greedy_selection(idsorted, min_phys=5, min_imgs=25)
        rest = idsorted
    elif num_of_imgs < 5000:
        part = greedy_selection(idsorted, min_phys=5, min_imgs=0.2 * num_of_imgs)
        rest = idsorted
    else:
        part = greedy_selection(idsorted, min_phys=5, min_imgs=1000)
        rest = idsorted

    part = unsort_idsorted(part)
    rest = unsort_idsorted(rest)

    return part, rest


# TODO add parameters for thresholds
# splits all classes using thresholds for physical signs and sign images count
def threshold_split(total):
    total_part, total_rest = {}, {}

    for class_name in sorted(total):
        class_part, class_rest = tsplit_class(total[class_name])
        # print(len(total[class_name]), len(class_part), len(class_rest))
        # if len(class_part) > 0:
        total_part[class_name] = class_part
        # if len(class_rest) > 0: # if there is something in class
        total_rest[class_name] = class_rest

    return total_rest, total_part

    

# def split(classes, test_ratio=0.2, val_ratio=0.):
    # print("splitting: test/rest")
#     # test, rest = split_proportionally(classes, test_ratio)
#     # test, rest = threshold_split(classes)
#     print("splitting: val/train")
#     val, train = split_proportionally(rest, val_ratio)
#
#     return train, test, val

        


#converts initial labelling (by images) to labelling organized by classes
def organize_by_classes(marking):
    res = {}
    for pict_name in sorted(marking):
        for sign in marking[pict_name]:
            sign['pict_name'] = pict_name
            #strip: "unknown_unmarked" should be equal "unknown_unmarked " (occures in original marking)
            res.setdefault(sign["sign_class"].strip(), []).append(sign)
    print("Number of classes : {}".format(len(sorted(res))))
    return res


def remove_empty(selection):
    classes = []
    for class_name in selection:
        if len(selection[class_name]) == 0:
            classes.append(class_name)

    for class_name in classes:
        del selection[class_name]

def intersect_by_classes(selection_1, selection_2):
    remove_empty(selection_1)
    remove_empty(selection_2)
    common_classes = set(selection_1).intersection(set(selection_2))
    print("in common: ", len(common_classes))
    selection_1 = { k:selection_1[k] for k in common_classes}
    selection_2 = {k: selection_2[k] for k in common_classes}

    return selection_1, selection_2


def raw_split(path):
    total_marking = load_marking("{path}/new_marking.json".format(**locals()))
    organized = organize_by_classes(total_marking)
    train, test = threshold_split(organized)
    marking = {'train': train, 'test': test}
    save_marking(marking, path, prefix="marking")



def get_group_id(class_id):
    if "1_1" <= class_id <= "1_8" or "2_3" <= class_id <= "2_4":
        return "red_triangles"

    elif "2_1" <= class_id <= "2_2":
        return "main_road"

    elif "2_6" == class_id or "3_10" <= class_id <= "3_20" or \
            "3_24_n" <= class_id <= "3_24_n80" or "3_32" <= class_id <= "3_7":
        return "red_circle"

    elif "2_7" == class_id or "5_11" <= class_id <= "6_15_3" or \
            "6_2_n" <= class_id <= "7_7":
        return "blue_rectangle"

    elif "2_5" == class_id or "3_1" == class_id:
       return "fully_red"

    elif "3_21" == class_id or "3_31" == class_id or \
            "3_25_n" <= class_id <= "3_25_n80":
        return "grey_circle"

    elif "3_27" <= class_id <= "3_30":
        return "blue_and_red_circle"

    elif "4_1_1" <= class_id <= "4_5":
        return "blue_circle"

    elif "8_13" <= class_id <= "8_8" or "6_16" == class_id or \
            "4_8_2" == class_id or "4_8_3" == class_id:
        return "white_rectangle"

    else:
        print("WARNING: unknown class id: ", class_id)
        return "UNKNOWN"


def set_class_id(bboxes, id):
    for bbox in bboxes:
        bbox["sign_class"] = id
    return bboxes


def group_classes(class_marking):
    groupped = {}
    for class_id in class_marking:
        group_id = get_group_id(class_id)

        groupped.setdefault(group_id, []).extend(
            set_class_id(class_marking[class_id], group_id))

    print("Number of classes after groupping: ", len(groupped))
    for id in groupped:
        print(id, " ", len(groupped[id]))
    exit(0)
    return  groupped


def classification_marking(path, group=False, seed=42, test_ratio=0.2, val_ratio=0.0):
    raw_split(path)

    total_marking = load_marking("{path}/new_marking.json".format(**locals()))
    no_ignored = remove_ignored(total_marking)
    organized = organize_by_classes(no_ignored)
    known = remove_unknown(organized)

    if group:
        known = group_classes(known)

    train, test = threshold_split(known)
    train, test = intersect_by_classes(train, test)
    train, val = val_split(train, ratio=val_ratio)
    marking = {'train': train, 'val': val, 'test': test}

    return marking



# def grouppedClassMarking(path, test_ratio=0.0, val_ratio=0.0):
#     total_marking = load_marking("{path}/new_marking.json".format(**locals()))
#     no_ignored = remove_ignored(total_marking)
#     organized = organize_by_classes(no_ignored)
#     known = remove_unknown(organized)
#
#     groupped = group_classes(known)



#previous version (when latest marking was spread across 2 files for train and test)
# def classification_marking(path, threshold=100, seed=42, test_ratio=0.2, val_ratio=0.1):
#     random.seed(seed)
#     gathered = gather_signs(path) #gather signs from 2 markings into one dictionary (key = sign_class)
#     print("Number of classes : {}".format(len(sorted(gathered))))
#     reduced = remove_small(remove_unknown(gathered), threshold)
#
#     train, test, val = split(reduced, test_ratio=test_ratio, val_ratio=val_ratio)
#     marking = {'train': train, 'val': val, 'test': test}
#     return marking



def save_marking(marking, path, prefix):
    print('saving markings')
    for phase in sorted(marking):
        filename = "{}/{}_{}.json".format(path, prefix, phase)
        with open(filename, 'w') as f:
            content = json.dumps(marking[phase], indent=4, sort_keys=True)
            f.write(content)



    

    