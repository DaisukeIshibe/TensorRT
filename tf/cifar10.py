# Build a model for CIFAR-10 dataset with simple VGG architecture in Keras
# This code conducts both train and evaluate the model using TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, models

def create_vgg_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def compile_model(model):
	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	return model

def get_cifar10_model(input_shape=(32, 32, 3), num_classes=10):
	model = create_vgg_model(input_shape, num_classes)
	model = compile_model(model)
	return model

def train_model(model, train_data, train_labels, epochs=10, batch_size=64):
	model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

def evaluate_model(model, test_data, test_labels):
	test_loss, test_accuracy = model.evaluate(test_data, test_labels)
	print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
	return test_loss, test_accuracy

def predict_with_saved_model(model_path, test_data):
	"""SavedModelを使って予測を実行"""
	model = tf.keras.models.load_model(model_path)
	predictions = model.predict(test_data)
	return predictions

def save_test_samples(x_test, y_test, num_samples=10):
	"""テスト用サンプルを保存"""
	import numpy as np
	
	# 最初のnum_samples個のサンプルを保存
	test_samples = x_test[:num_samples]
	test_labels = y_test[:num_samples]
	
	np.save('test_samples.npy', test_samples)
	np.save('test_labels.npy', test_labels)
	
	print(f"Saved {num_samples} test samples to test_samples.npy and test_labels.npy")
	return test_samples, test_labels

# Example usage:
if __name__ == "__main__":
	# Load CIFAR-10 dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	
	# Normalize the data
	x_train, x_test = x_train / 255.0, x_test / 255.0
	
	# Create and compile the model
	model = get_cifar10_model()
	
	# Train the model
	train_model(model, x_train, y_train, epochs=5)  # Reduce epochs for faster testing
	
	# Evaluate the model
	test_loss, test_accuracy = evaluate_model(model, x_test, y_test)

	# Save the model with SavedModel format
	model.save('cifar10_vgg_model')
	print("Model saved to 'cifar10_vgg_model'")
	
	# Save test samples for consistency check
	test_samples, test_labels = save_test_samples(x_test, y_test, 10)
	
	# Test SavedModel prediction
	saved_predictions = predict_with_saved_model('cifar10_vgg_model', test_samples)
	original_predictions = model.predict(test_samples)
	
	# Verify consistency
	import numpy as np
	consistency_check = np.allclose(saved_predictions, original_predictions, rtol=1e-5)
	print(f"SavedModel consistency check: {'PASSED' if consistency_check else 'FAILED'}")
	
	# Save predictions for later comparison
	np.save('savedmodel_predictions.npy', saved_predictions)
	print("SavedModel predictions saved to 'savedmodel_predictions.npy'")

# Note: Ensure you have TensorFlow installed in your environment to run this code.