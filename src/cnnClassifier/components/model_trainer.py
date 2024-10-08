import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path

from PIL import Image
import scipy

from pathlib import Path
import tensorflow as tf

from cnnClassifier.entity.config_entity import (TrainingConfig)

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        # Load the base model from the specified path using forward slashes or raw string
        base_model_path = Path(r'C:/Users/Swarnendu/Desktop/End-to-end-Chest-Cancer-Classification-using-MLflow-DVC/artifacts/prepare_base_model/base_model.h5')
        base_model = tf.keras.models.load_model(base_model_path)

        # Add new layers to the model
        x = base_model.output
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(2, activation='softmax')(x)

        # Create a new model
        self.model = tf.keras.Model(inputs=base_model.input, outputs=x)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )


    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Ensure the directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.with_suffix('.keras')
        model.save(path)


    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
