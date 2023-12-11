
# Rock Paper Scissors Prediction

Ini merupakan sebuah web simple dengan tujuan untuk memprediksi gambar dari tangan yang menggambarkan batu, gunting dan kertas.

Dataset yang digunakan : [Dataset Batu Gunting Kertas](https://drive.google.com/file/d/1X9jFokn9AXMMVTmlBQ7XZpBsLKVFnp-d/view?usp=drive_link)

Model yang digunakan : Convolution2D

Akurasi yang didapatkan adalah : 99.8% dengan loss sebesar : 0.2%

![Convolution2D](https://raw.githubusercontent.com/Ilham-AM462/rps-modul6/main/Screenshot_20231212_000229.png)


## Preprocessing

Preprocessing : 

```py
trainimg_gen =  ImageDataGenerator(rescale=(1.0/255),
                              zoom_range=0.2,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              fill_mode='nearest')

img_gen = trainimg_gen =  ImageDataGenerator(rescale=(1.0/255))

train_gen = trainimg_gen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

val_gen = img_gen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)
```

Model : 

```py
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])
```

Summary Model : 
![Summary Model](https://raw.githubusercontent.com/Ilham-AM462/rps-modul6/main/Screenshot_20231212_000229.png)

Accuracy plot : 

![Model Accuracy Graph](https://raw.githubusercontent.com/Ilham-AM462/rps-modul6/main/Screenshot_20231212_001726.png)

Loss plot : 

![Model Loss Graph](https://raw.githubusercontent.com/Ilham-AM462/rps-modul6/main/Screenshot_20231212_001733.png)
## Local Deployment

Result on Local Deployment : 
![Local Deployment](https://raw.githubusercontent.com/Ilham-AM462/rps-modul6/main/Screenshot_20231211_235649.png)
## Authors

- [@Ilham-AM462](https://github.com/Ilham-AM462)

