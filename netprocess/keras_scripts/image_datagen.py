from keras.preprocessing.image import ImageDataGenerator


def init_generator(config, phase):
    if phase == 'train':
        return ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
    #                 width_shift_range=0.125,
    #                 height_shift_range=0.125,
                    horizontal_flip=False,
                    vertical_flip=False)
    else:
        return ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    horizontal_flip=False,
                    vertical_flip=False)



def image_generator(config, phase, test_batch_size=1):
    datagen = init_generator(config, phase)
    root = config['img']['processed_path']
    directory = '{root}/{phase}'.format(**locals())
    size = config['img']['img_size'] + 2 * config['img']['padding']
    
    if phase == 'train':
        shuffle = True
        batch_size = config['train_params']['batch_size']
    else: 
        shuffle = False
        print('batch_size for {} = {}'.format(phase, test_batch_size))
        batch_size = test_batch_size
    
    generator = datagen.flow_from_directory(
            directory,  
            target_size=(size, size),
            batch_size=batch_size,
            class_mode='categorical',
            save_to_dir=None, 
            shuffle=shuffle)
    
    return generator
