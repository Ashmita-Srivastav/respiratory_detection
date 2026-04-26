from respiratory_disease_predictor import CoswaraDatasetLoader, RespiratoryDiseaseTrainer

if __name__ == "__main__":
    print("Initializing dataset loader...")
    loader = CoswaraDatasetLoader()
    
    print("Preparing dataset (this may take a while)...")
    X_flat, X_cnn, y = loader.prepare_dataset(use_augmentation=True)
    
    print("Initializing trainer...")
    trainer = RespiratoryDiseaseTrainer()
    
    print("Starting training process...")
    trainer.train(X_cnn, y, model_type='cnn_lstm')
    
    print("Training complete and models saved successfully!")
