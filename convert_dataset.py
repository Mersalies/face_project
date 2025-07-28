from obrabotka_face import process_face_dataset


RAW_DIR_TRAIN = "archive/train"                
OUT_DIR_TRAIN = "processed_dataset/train"      


RAW_DIR_VAL = "archive/val"                # папка с оригинальными изображениями
OUT_DIR_VAL = "processed_dataset/val"      # папка с обработаными изображениями

process_face_dataset(raw_dir= RAW_DIR_VAL, out_dir=OUT_DIR_VAL)