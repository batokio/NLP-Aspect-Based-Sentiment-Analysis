from random import shuffle
from imports import *
from preprocessing import *
from model import *
from utils import *



class Classifier:
    """The Classifier"""

    #############################################
    def train(self, trainfile, devfile=None):
        """
        Trains the classifier model on the training set stored in file trainfile
        WARNING: DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        
        # Load the files into pandas dataframe
        trainfile = pd.read_csv(trainfile, sep='\t', header=None)
        if devfile is not None:
            devfile = pd.read_csv(devfile, sep='\t', header=None)
        
        self.n_classes = len(trainfile[0].unique())
        
        # Set the batch size
        self.bs = 32
        
        # Proprocess the file
        self.trainfile = preprocessing(trainfile)
        
        if devfile is not None:
            self.devfile = preprocessing(devfile)
            
        self.n_classes = len(self.trainfile[0].unique())
        
        # Set the pretrained model
        # Case-sensitive BERT model
        PRE_TRAINED_MODEL = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
        
        # Set the maximum sequence length for the encoding of the sentences
        self.max_length = sequence_length(self.trainfile, self.tokenizer)
        
        # Load the dataloader
        self.train_loader = create_dataloader(self.trainfile, self.tokenizer, self.max_length, self.bs)
        
        if devfile is not None: 
            self.val_loader = create_dataloader(self.devfile, self.tokenizer, self.max_length, self.bs)
            
        # Define the devide and the model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = SentimentClassifier(self.n_classes)
        self.model = self.model.to(self.device)
        
        # Hyperparameters
        self.epochs = 10
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        self.total_steps = len(self.train_loader) * self.epochs
        
        self.scheduler = lr_scheduler.LinearLR(
            self.optimizer,
            total_iters=self.total_steps
            )
        
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        
        ## Training loop
        self.training_loop()

        
        

        
    def training_loop(self):
        
        history = defaultdict(list)
        best_accuracy = 0
        
        for epoch in tqdm(range(self.epochs)):
            
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)
            
            train_acc, train_loss = self.training_func(
                self.model,
                self.train_loader,
                self.loss_fn,
                self.optimizer,
                self.device,
                self.scheduler,
                len(self.trainfile)
                )
            
            print(f'Train loss {train_loss} accuracy {train_acc}')
            
            if self.val_loader is not None:
                    
                val_acc, val_loss = self.eval_func(
                    self.model,
                    self.val_loader,
                    self.loss_fn,
                    self.device,
                    len(self.devfile)
                    )

                print(f'Val loss {val_loss} accuracy {val_acc}')
                print()
                
            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
                
            if self.val_loader is not None:
                history['val_acc'].append(val_acc)
                history['val_loss'].append(val_loss)
                
            if (self.val_loader is not None) and (val_acc > best_accuracy):
                torch.save(self.model.state_dict(), 'best_model_state.pth')
                best_accuracy = val_acc
            
        
    # Helper function for the training of one epoch
    def training_func(
        self,
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
        ):
        
        model.train()
        
        losses = []
        correct_predictions = 0
        
        # Iterate over the batch
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            
            # Compute the output of the model
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            # Backprop
            loss.backward()
            
            # Gradient clipping to mitigate the problem of exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        return correct_predictions.double() / n_examples, np.mean(losses)
    
    
    # Helper function for the training of one epoch
    def eval_func(self, model, data_loader, loss_fn, device, n_examples):
        
        model = model.eval()    
        losses = []
        correct_predictions = 0
        
        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                labels = d["labels"].to(device)
                
                # Compute the output of the model
                outputs = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )
                
                _, preds = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)
            
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                
        return correct_predictions.double() / n_examples, np.mean(losses)
        


    def predict(self, datafile):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        """
        
        # Load the files into pandas dataframe
        datafile = pd.read_csv(datafile, sep='\t', header=None)
        
        # Preprocess the file
        datafile = preprocessing(datafile)
        
        # Get the dataloader
        data_loader = create_dataloader(datafile, self.tokenizer, self.max_length, self.bs, shuffle=False)
        
        
        # Load the best model
        self.model = SentimentClassifier(self.n_classes).to(self.device)
        # self.model.load_state_dict(torch.load('best_model_state.pth', map_location = 'cpu'))
        self.model.load_state_dict(torch.load('best_model_state.pth'))

        
        # Set the model to evaluation mode
        self.model.eval()
        
        output_labels = []

        for d in data_loader:
            
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)
            

            with torch.no_grad():
                
                outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                            )
                

            logits = F.softmax(outputs, dim=1)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1)

            # converting integer labels into named labels
            for label in outputs:
                if label == 2:
                    output_labels.append('positive')
                
                elif label == 1:
                    output_labels.append('neutral')
                    
                elif label == 0:
                    output_labels.append('negative')

        return np.array(output_labels)
                
                
                
        