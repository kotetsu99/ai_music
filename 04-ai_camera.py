#!/usr/bin/env python
# -*- coding: utf-8 -*-

import keras
import numpy as np
import cv2
import picamera
import picamera.array
import os, sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

# URL定数定義
PLAYLIST_URL = 'https://music.youtube.com/watch?v=0cGBTbjvwuo&list=RDAMVM0cGBTbjvwuo'
WEB_DRIVER = '/usr/bin/chromedriver'

# プログラム実行制限時間(分)
time_limit = 300
# 本人の名前
person = 'person'

# OpenCV物体検出サイズ定義
cv_width, cv_height = 100, 100
# OpenCV物体検出閾値
minN = 4
# 顔画像サイズ定義
img_width, img_height = 64, 64
# 顔検出用カスケードxmlファイルパス定義
cascade_xml = "haarcascade_frontalface_alt.xml"

# 学習用データセットのディレクトリパス
train_data_dir = 'dataset/02-face'
# データセットのサブディレクトリ名（クラス名）を取得
#classes = os.listdir(train_data_dir)
classes = ('others', 'person')


def main():

    # 環境設定(ディスプレイの出力先をlocalhostにする)
    os.environ['DISPLAY'] = ':0'
    print('クラス名リスト = ', classes)

    # 学習済ファイルの確認
    if len(sys.argv)==1:
        print('使用法: python 本ファイル名.py 学習済ファイル名.h5')
        sys.exit()
    savefile = sys.argv[1]

    # モデルのロード
    model = keras.models.load_model(savefile)

    # ブラウザを起動
    brws = setup_browser()

    print('顔認識を開始')

    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            # カメラの解像度を320x320にセット
            camera.resolution = (320, 320)
            # カメラのフレームレートを15fpsにセット
            camera.framerate = 15
            # ホワイトバランスをfluorescent(蛍光灯)モードにセット
            camera.awb_mode = 'fluorescent'

            # 時間計測開始
            start_time = time.time()
            process_time = 0

            # プレイヤー状態初期値設定
            state = 'None'

            # 制限時間まで顔認識実行
            while process_time < time_limit :
                # 本人認識フラグ初期化
                person_flg = False
                # プレイヤーの状態を確認
                state = check_player_state(brws)
                #print(state)
                # stream.arrayにBGRの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)

                # 顔認識
                image, person_flg = detect_face(stream.array, model, person_flg)
                # カメラ映像をウインドウに表示
                cv2.imshow('frame', image)

                # 'q'を入力でアプリケーション終了
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                # 本人が顔検出された場合
                if (person_flg == True):
                    # プレイヤー画面が開いていない場合、プレイヤー画面に遷移
                    if state == 'None':
                        brws.get(PLAYLIST_URL)
                        player = brws.find_element_by_tag_name('body')
                        print('再生')
                    # プレイヤー画面が開いている場合、再生状態とする
                    elif state == 'Pause':
                        # プレイヤー再生
                        player.send_keys(Keys.SPACE)
                        print('再生')
                # 本人が顔検出されない場合
                elif (person_flg == False):
                    # プレイヤーが再生状態の場合
                    if state == 'Play':
                        # プレイヤー一時停止
                        player.send_keys(Keys.SPACE)
                        print('一時停止')

                # streamをリセット
                stream.seek(0)
                stream.truncate()

                # 経過時間(分)計算
                process_time = (time.time() - start_time) / 60
                #print('process_time = ', process_time, '[min]')

            cv2.destroyAllWindows()


def detect_face(image, model, person_flg):
    # グレースケール画像に変換
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_xml)

    # 顔検出の実行
    face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=minN,minSize=(cv_width, cv_height))

    # 顔が1つ以上検出された場合
    if len(face_list) > 0:
        for rect in face_list:
            # 顔画像を生成
            face_img = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            if face_img.shape[0] < cv_width or face_img.shape[1] < cv_height:
                #print("too small")
                continue
            # 顔画像とサイズを定義
            face_img = cv2.resize(face_img, (img_width, img_height))

            # Keras向けにBGR->RGB変換、float型変換
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            # 顔画像をAIに認識
            name = predict_who(face_img, model)
            #print(name)
            # 顔近傍に矩形描画
            cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness = 3)
            # AIの認識結果(人物名)を元画像に矩形付きで表示
            x, y, width, height = rect
            cv2.putText(image, name, (x, y + height + 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255) ,2)
            # 画像保存
            #cv2.imwrite(name + '.jpg', image)
            if name == person :
                person_flg = True
    return image, person_flg


def predict_who(x, model):
    # 画像データをテンソル整形
    x = np.expand_dims(x, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    x = x / 255
    pred = model.predict(x)[0]

    #print(pred)
    # 本人:1 他人:0のようにラベル付けされている。
    result = pred[0]
    print('顔認識：本人である確率=' + str(100 * result) + '[%]')

    # 50%を超えていれば本人と判定。50%以下は他人と判定。
    if result > 0.5:
        name = classes[1]
    else:
        name = classes[0]

    # 1番予測確率が高い人物を返す
    return name


def setup_browser():
    # ブラウザ起動オプション設定
    options = webdriver.ChromeOptions()
    options.add_argument('--kiosk')
    options.add_argument('--incognito')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-extensions')

    # ブラウザ起動
    brws = webdriver.Chrome(WEB_DRIVER, chrome_options=options)
    brws.set_page_load_timeout(90)

    return brws


def check_player_state(brws):

    if len(brws.find_elements_by_id('play-pause-button')) > 0 :
        player_element = brws.find_element_by_id('play-pause-button')
        title = player_element.get_attribute('title')
        # プレイヤーの再生状態に応じてstateを変更
        if title == '一時停止':
            state = 'Play'
        elif title == '再生':
            state = 'Pause'
    else:
        # stateは'None'とする
        state = 'None'

    return state


if __name__ == '__main__':
    main()
