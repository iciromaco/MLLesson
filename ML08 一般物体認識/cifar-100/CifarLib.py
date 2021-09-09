from tensorflow.keras.utils import plot_model, to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib import cm
import math
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger,TensorBoard

labelsJ = np.array(['リンゴ', '観賞魚', '赤ちゃん', 'クマ', 'ビーバー', 'ベッド', '蜂', '甲虫', '自転車', 'ボトル', 
       'ボウル', '男の子', '橋', 'バス', '蝶', 'ヒツジ','缶', '城', 'いも虫', 'ウシ', 
       '椅子','チンパンジー','時計', '雲', 'ゴキブリ', 'ソファー', 'カニ', 'クロコダイル', 'カップ', '恐竜', 
       'イルカ', 'ゾウ', 'ヒラメ', '森林', 'キツネ', '女の子', 'ハムスター', '家', 'カンガルー', 'キーボード', 
       'ランプ', '芝刈り機', 'ヒョウ', 'ライオン', 'トカゲ','ロブスター', '男性', 'カエデ', 'オートバイ', '山', 
       'マウス', 'キノコ', 'カシ', 'オレンジ', 'ラン','カワウソ', 'ヤシ', 'ナシ', 'ピックアップトラック', 'マツ', 
       '平原', 'プレート', 'ポピー', 'ヤマアラシ', 'フクロネズミ', 'ウサギ', 'ラクーン', 'エイ', '道路','ロケット', 
       'バラ', '海', 'アザラシ', 'サメ','トガリネズミ', 'スカンク', '超高層ビル', 'カタツムリ', 'ヘビ', 'クモ', 
       'リス', '路面電車', 'ヒマワリ', 'ピーマン', 'テーブル', 'タンク', '電話', 'テレビ', 'トラ', 'トラクター', 
       '列車', 'マス', 'チューリップ', 'カメ', 'タンス', 'クジラ', 'ヤナギ', 'オオカミ', '女性', 'ミミズ'])

labels = np.array(['apple', 'aquarium_fish', 'baby', 'bear', 'beaver','bed','bee','beetle','bicycle','bottle',
                   'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
                   'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
                   'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
                   'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
                   'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
                   'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
                   'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
                   'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
                   'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'])

SIZE = 32
CLASSES = 3 # カテゴリ数
DATASIZE = SIZE *  SIZE * 3

class NNModel():
  def __init__(self,model,catlist,dset = None,type='CNN'):
      self.model = model # ニューラルネットワークのモデル定義
      Xtrain0,ytrain,Xtest0,ytest = dset # データセット
      self.Xtrain0,self.Xtest0 = Xtrain0,Xtest0
      self.CATLIST = catlist # カテゴリリスト
      if type == 'CNN': # 畳み込みニューラルネットワークの場合
        self.Xtrain = Xtrain0
        self.Xtest = Xtest0
      else: # type == 'MLP'フラット入力のネットワークの場合
        self.Xtrain = Xtrain0.reshape(len(Xtrain0),DATASIZE)
        self.Xtest = Xtest0.reshape(len(Xtest0),DATASIZE)
      self.ytrain = ytrain
      self.ytest = ytest
      self.TrainError,self.TestError = [],[]
  def summary(self):
      self.model.summary()
  def plot(self,fname):
      plot_model(self.model,to_file=fname)
  def compile(self,lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0):
      self.model.compile(loss='sparse_categorical_crossentropy',
          optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay),
          metrics=['accuracy'])
  def learn(self,withCompile=True,verbose=1,epochs=100):
      if withCompile:
        self.compile()
      es = EarlyStopping(monitor='loss', patience=5)   #  訓練用データのロスが改善されなくなったら2エポック後に停止
      tb_cb = TensorBoard(log_dir='tblog', histogram_freq=1, write_graph=True)
      csv_logger = CSVLogger('training.log')
      self.hist = self.model.fit(self.Xtrain ,self.ytrain,
                  epochs=epochs,
                  verbose=verbose,
                  callbacks=[es, csv_logger])
  # 学習過程のグラフ化
  def hplot(self):
      fig, ax1 = plt.subplots()
      ax2 = ax1.twinx()
      acc = self.hist.history['accuracy']
      loss = self.hist.history['loss']
      ax1.plot(range(1, len(loss)+1), loss,color=cm.Set1.colors[0],label='loss') # 誤差
      ax2.plot(range(1, len(acc)+1), acc,color=cm.Set1.colors[1],label='accuracy') #  正解率
      ax1.set_ylabel('Loss')
      ax2.set_xlabel('Epochs')
      ax2.set_ylabel('Accuracy')
      ax1.legend()
      ax2.legend()
      plt.show()
  def recognitionResult(self,mode='all'): # all, train, test
      if mode == 'all':
        self.recognitionResult(mode='train')
        self.recognitionResult(mode='test')
      else:
        if mode == 'test':
          Xdata = self.Xtest
          ydata = self.ytest
          print("\n\033[1m\033[35mテストデータの認識率\033[0m")
        elif mode == 'train':
          Xdata = self.Xtrain
          ydata = self.ytrain
          print("\033[1m\033[35m訓練データの認識率\033[0m")
        # 訓練データに対する識別結果
        ndata = len(Xdata) # データ数
        predictT = self.model.predict(Xdata)
        predictT = [np.argmax(n1)  for n1 in predictT]
        NCAT = len(self.CATLIST) # カテゴリ数
        ct1 = np.zeros((NCAT,NCAT),np.uint16) # 認識結果集計表
        Error = []
        for i in range(ndata):
            ct1[int(ydata[i]),int(predictT[i])] += 1
            if ydata[i] != predictT[i]:
                Error.append([i,int(ydata[i]),int(predictT[i])])
        print("誤認識 {0:}/{1:} \n　正答率={2:5.1f}　誤り率＝{3:5.1f} %\n".format(len(Error),len(ydata),100*(ndata-len(Error))/ndata,100*len(Error)/ndata))
        print("カテゴリごとの認識結果と正答率")
        catlist = self.CATLIST
        crossT = pd.concat([pd.DataFrame(catlist,columns=['正解カテゴリ']),pd.DataFrame(ct1,columns=catlist)],axis=1)
        crossT = pd.concat([crossT,pd.DataFrame([np.round(1000*crossT[cat][i]/ndata*NCAT)/10 for i,cat in enumerate(catlist)],columns=['正答率'])],axis=1).set_index('正解カテゴリ')
        display(crossT.head())
        if mode == 'train':
          self.trainError = Error
        else:
          self.testError = Error

  # 誤認識した画像を表示
  def showErrorImages(self,mode = 'test'):
      errlist = self.testError if mode == 'test' else self.trainError
      images = self.Xtest0 if mode == 'test' else self.Xtrain0

      CATLIST = self.CATLIST
      # 認識間違いの表示
      def showEimg(samples):
          last = len(samples) 
          plt.figure(figsize=(8,7.5*(math.ceil(last/8))/6),dpi=100)
          for i in range(last):
                  plt.subplot((last-1)//8+1,8,i+1)
                  plt.xticks([])
                  plt.yticks([])
                  plt.imshow(images[samples[i][0]])
                  plt.title("{}\n →{}".format(CATLIST[samples[i][1]],CATLIST[samples[i][2]]),fontsize=6)

      if len(errlist)>0:
        samples = errlist
        if len(errlist)>16:
          print("下に示す例を含め",len(errlist),"枚")
          samples = [errlist[int(x)] for x in np.linspace(0,len(errlist)-1,16)]
        showEimg(samples)
      else:
        print("誤認識はありません")

# データ・セットの start 番目から最大100画像分を表示
def showimages(images, start=0):
    canvas = Image.new('RGB',(350,350),(255,255,255))
    dsize = len(images)
    end = start + 100
    if start + 100 > dsize:
        end = dsize
    for i in range(10):
        for j in range(10):
            n = start+i*10+j
            if n >= end:
                break
            else:
                img = Image.fromarray(images[start+i*10+j])
                canvas.paste(img,(35*j,35*i))
    return canvas

# 名称からカテゴリid を調べる
def word2fcat(word):
    return np.where(labels==word)[0][0]
def word2fcatJ(word):
    return np.where(labelsJ==word)[0][0]

# CIFER-100のデータ data から、カテゴリ名 cat の画像だけ抽出する 
def getCatN(X,y,n):
    collections = (y==[n]).flatten()
    images = X[collections]
    return images

# カテゴリの英名 cat の画像だけ抽出する  
def getCatE(X,y,cat):
    return getCatN(X,y,word2fcat(cat))

# カテゴリの和名 cat の画像だけ抽出する  
def getCatJ(X,y,cat):
    return getCatN(X,y,word2fcatJ(cat))
