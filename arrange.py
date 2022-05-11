import os
import pandas as pd
import shutil
from PIL import Image

if __name__ == "__main__":

    train_path = "./data/train/"
    test_path = "./data/test/"
    valid_path = "./data/valid/"

    train_folder = os.listdir(train_path)
    test_folder = os.listdir(test_path)
    valid_folder = os.listdir(valid_path)

    dis_train_path = "./dataset/train/"
    dis_test_path = "./dataset/test/"
    dis_valid_path = "./dataset/valid/"

    if not os.path.exists(dis_train_path):
        os.makedirs(dis_train_path)
    if not os.path.exists(dis_test_path):
        os.makedirs(dis_test_path)
    if not os.path.exists(dis_valid_path):
        os.makedirs(dis_valid_path)


    if train_folder == test_folder == valid_folder :
        class_names = []
        img_names = []
        index=0
        for folder_name in train_folder:
            folder_path = train_path + folder_name
            folder = os.listdir(folder_path)
            for image_name in folder:
                try:
                    image = Image.open(os.path.join(folder_path, image_name))
                    if image.mode=='RGB':
                        dst_path = dis_train_path+ folder_name+ "_"+ image_name[:-4]+".jpg"
                        shutil.copy(os.path.join(folder_path, image_name), dst_path)
                        class_names.append(index)
                        img_names.append(folder_name+"_"+image_name[:-4]+".jpg")
                except:
                    pass
            index+=1

        data = {'class': class_names, 'image':img_names}
        train_df = pd.DataFrame(data=data)      
        train_df.to_csv("./dataset/train.csv",index=False)

        class_names = []
        img_names = []
        index=0
        for folder_name in test_folder:
            folder_path = test_path + folder_name
            folder = os.listdir(folder_path)
            for image_name in folder:
                try:
                    image = Image.open(os.path.join(folder_path, image_name))
                    if image.mode=='RGB':
                        dst_path = dis_test_path+ folder_name+ "_"+ image_name[:-4]+".jpg"
                        shutil.copy(os.path.join(folder_path, image_name), dst_path)
                        class_names.append(index)
                        img_names.append(folder_name+"_"+image_name[:-4]+".jpg")
                except:
                    pass
            index+=1
        data = {'class': class_names, 'image':img_names}
        test_df = pd.DataFrame(data=data)      
        test_df.to_csv("./dataset/test.csv",index=False)    
        
        
        class_names = []
        img_names = []
        index=0
        for folder_name in valid_folder:
            folder_path = valid_path + folder_name
            folder = os.listdir(folder_path)
            for image_name in folder:
                try:
                    image = Image.open(os.path.join(folder_path, image_name))
                    if image.mode=='RGB':
                        dst_path = dis_valid_path+ folder_name+ "_"+ image_name[:-4]+".jpg"
                        shutil.copy(os.path.join(folder_path, image_name), dst_path)
                        class_names.append(index)
                        img_names.append(folder_name+"_"+image_name[:-4]+".jpg")
                except:
                    pass  
            index+=1
        data = {'class': class_names, 'image':img_names}
        valid_df = pd.DataFrame(data=data)      
        valid_df.to_csv("./dataset/valid.csv",index=False) 

    print("done!")








