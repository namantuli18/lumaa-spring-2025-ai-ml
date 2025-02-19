# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST
---

## Model Training

1. **Download the dataset from kaggle**
   
   ```bash
   curl -L -o mpst-movie-plot-synopses-with-tags.zip https://www.kaggle.com/api/v1/datasets/download/cryptexcode/mpst-movie-plot-synopses-with-tags
   ```
2. **Unzip the solution to unpack the csv files**

   ```bash
   unzip mpst-movie-plot-synopses-with-tags.zip
   ```
3. **Install the requirements**

   ```python
   pip install -r requirements.txt
   ```
4. **Follow along the steps in the notebook to visualize data, train, and save the model for inference**
   [Notebook](https://github.com/namantuli18/lumaa-spring-2025-ai-ml/blob/main/train_model.ipynb)

---

### [Video Demo](https://drive.google.com/file/d/1fwsEQ6SePHa9b7eEmvKa_R3X0mQkV9YM/view?usp=sharing)

## Model Inference

1. **Using python notebook**  
   - Just run the cell that calls the function recommend_movies in the [notebook](https://github.com/namantuli18/lumaa-spring-2025-ai-ml/blob/main/train_model.ipynb)
   - Sample request and response
     ```python
     
     user_query = "I like action movies set in space."
     recommend_movies(user_query)

     ```

     ```
     User Input: I like action movies set in space.
     Top Movie Recommendations:
     
     1. Future War 198X (Similarity: 0.1171)
     2. Gravity (Similarity: 0.1042)
     3. The Right Stuff (Similarity: 0.1011)
     4. Power Rangers Lost Galaxy (Similarity: 0.1004)
     5. Duke Nukem 3D (Similarity: 0.0927)

     ```

2. **Using python script and command line args**  
   - Call the python script that loads the model and sends the prediction in response using the script
     ```bash
        lumaa-spring-2025-ai-ml> python recommend.py "A detective solving a mysterious murder in a small town"
     ```
     Response:
     ```
         User Input: A detective solving a mysterious murder in a small town
         
         Top Movie Recommendations:
         
         1. Untitled Murder in Paris Project (Similarity: 0.1267)
         2. Purgatory (Similarity: 0.1125)
         3. Exposed (Similarity: 0.1083)
     ``` 

3. **Use the app deployed on streamlit**
   ![Website](https://github.com/namantuli18/lumaa-spring-2025-ai-ml/blob/main/imgs/img.jpg)

   - Run the script below to deploy the model on a streamlit server:
     ```bash
     python -m streamlit run host.py
     ```
   - Enter the movie description in a prompt and get the list of recommendations!
   
