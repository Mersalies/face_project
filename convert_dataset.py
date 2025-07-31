from obrabotka_face import process_face_dataset, generation_embedding


RAW_DIR_TRAIN = "archive/train"                
OUT_DIR_TRAIN = "processed_dataset/train"      


RAW_DIR_VAL = "archive/val"                # папка с оригинальными изображениями
OUT_DIR_VAL = "processed_dataset/val"      # папка с обработаными изображениями
# process_face_dataset(raw_dir= RAW_DIR_VAL, out_dir=OUT_DIR_VAL)


RAW_DIR_VER = "photo"
OUR_DIR_VER = "photo_to_verifecation/MyPhoto"
# process_face_dataset(raw_dir=RAW_DIR_VER, out_dir=OUR_DIR_VER)

RAW_DIR_EMD = "photo_to_verifecation/MyPhoto"
OUT_DIR_EMD = "photo_to_verifecation/embding_photo"
# generation_embedding(raw_dir=RAW_DIR_EMD, out_dir=OUT_DIR_EMD)

