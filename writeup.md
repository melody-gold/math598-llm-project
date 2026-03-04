# LLM Transformer WriteUp
========================== 

### Architecture Design Choices: 

The model has six configurable hyperparameters including d_model, d_vocab, d_hidden, n_layers, n_context, and n_context_max. The hyperparameter d_hidden will control the size of the multi-layer perceptron (MLP). The hyperparameter n_layers is used directly in the transformer block and sets depth of the model. Additionally, a hyperparameter of batchsize can be set when training of the model 
is called, but the parameter is not set within the configuration. 

The transformer architecture utilizes a decoder-only architecture to achieve text generation. Our team created a simple attention head and added positional embeddings to track word order. Prior to training, a function called Book_Dataset was created to tokenize the text from the book data and seperate it into sequences using n_context. The outputs from Book_Dataset are inputted directly into the training loop function and with the consistent n_context length, the model has increased computational efficiency. Within the training process, the model uses Cross 
Entropy loss with the AdamW optimizer. 

Training took around 15 minutes the first few times we ran it. As this was a lot of time, especially given our limited time together as a group, we have to reduce the context length and the "raw text" length for our data input. 

Overall we did a lot very well, most prominently, translating the math from lecture to real code as well as creating the tokenizer. Both of these seemed to come to us so easily as we all have a good math foundation whereas the compisci details, such as how embedding, encoding, and decoding should function, took longer to implement.  


### Challenges:

Generally, working on GitHub repositories in groups can be challenging as multiple people are editing the same file at once and our group was no exception. For example, when working on the project in class, our group faced challenges with merge conflicts as we would pull and push new edits. From this challenge, our group was able to work on coordinating code changes, dividing up work within the python journal, and keeping branches up to date with careful version control practices. 

During initial training of the model, our group ran into errors due to incompatible dimensions. 

Another challenge we faced figuring out how to filter online information. As we have a very specific project on our hands, doing research on a concept we were confused about was hard as some information were things our professor intentionally omitted from our project and understanding for the time being. 

One big challenge for us was writing pytest tests. This was the first time any of us had even heard of this idea so even knowing what kind of tests to make was hard and complicated. However, once we made the first 6 or so tests, the idea of tests became more clear and soon more followed that were a lot easier to make. 


### Future Directions: 

One direction the team explored but did not implement due to time constraints was a Layer Norm within the Multi-Layer Perceptron function. 


