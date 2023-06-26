from fastai.vision.all import *

def main():
    # Carga el modelo entrenado
    learn = load_learner('trained_model.pkl')

    # Lista de imágenes de prueba
    test_images = ['test/1.jpg', 'test/2.jpg', 'test/3.jpg', 'test/4.jpg', 'test/5.jpg',
                   'test/6.jpg', 'test/7.jpg', 'test/8.jpg', 'test/9.jpg', 'test/10.jpg']

    # Predice la clase de cada imagen de prueba
    for image_path in test_images:
        img = PILImage.create(image_path)
        pred,pred_idx,probs = learn.predict(img)
        print(f'La imagen {image_path} es de: {pred}, con una probabilidad de: {probs[pred_idx]:.04f}')

if __name__ == '__main__':
    main()