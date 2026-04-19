#!/usr/bin/env python
# coding: utf-8

# # Multimodal Graph Neural Networks for Movie Recommendation
# 
# ### **Assignment Overview**
# In this assignment, we will build a hybrid recommendation system using Computer Vision (CV), Natural Language Processing (NLP), and Graph Neural Networks (GNNs). You will complete three main tasks:
# 
# 1. **Task 1: Computer Vision (CV)** - You will tackle multi-label movie genre classification using two approaches: training a custom CNN from scratch and fine-tuning a pre-trained model on movie posters. After comparing their performance, you will extract dense visual embeddings from the penultimate layer of your best-performing model to use in the final recommendation graph.
# 2. **Task 2: Natural Language Processing (NLP)** - Train an RNN or Transformer on movie plot summaries to predict genres, extracting semantic text embeddings.
# 3. **Task 3: Graph Neural Networks (GNN)** - Construct a bipartite user-item graph using the `ml-latest-small` MovieLens dataset. You will initialize the movie nodes using the embeddings extracted from Tasks 1 & 2, and train a GNN to perform link prediction (recommending movies to users).
# 
# 

# ## ⚠️ Important Instructions
# 
# Before you begin:
# 
# 1. **Enable GPU**:
# Deep learning models (especially CNNs and sequence models) take hours to train on a standard CPU. You **must** enable a GPU for this assignment (either in Google Colab or Kaggle.
# 
# 2. **Training Time & Checkpointing**
# The default epochs in the skeleton code are kept low just to ensure your code runs without crashing. **To get good performance, you might need to increase the number of epochs.** If you train for many epochs, your Colab or Kaggle session might disconnect and you will lose your progress. Be prepared to implement **model checkpointing** (`torch.save(model.state_dict(), 'checkpoint.pth')`) to save your weights during training.
# 
# 3. **Adherence to Code Structure**:
# The datasets and data-loading pipelines provided in this notebook are specifically engineered so that the outputs of Task 1 and Task 2 seamlessly plug into Task 3.
# **Do not alter the predefined tensor shapes, variable names, or data splits.**
# 
# 4. **Grading**
# * **(Good):** Successfully completing the `TODO` sections, ensuring the code runs without errors, and writing a basic training loop will earn you a solid, passing grade.
# * **Full Points (Excellent):** To achieve full marks, you are expected to go beyond the skeleton code. This means implementing relevant **optimizations** (as instructed) and achieving better than baseline F1-scores.
# 
# ---

# ## Data Preparation
# This section contains the **mandatory data preparation phase**. Read through the explanations to understand *why* we are formatting the data this way before you begin building your models.
# 
# You can use the provided curated dataset for consistency, or recreate it yourself by running the following cells. Note that recreating the data requires a [TMDB API](https://developer.themoviedb.org/docs/getting-started) key.
# 
# 

# ---
# ### **Download the Raw Data**
# We need two datasets from GroupLens - [MovieLens](https://grouplens.org/datasets/movielens/):
# * `ml-latest-small`: A small bipartite graph of 600 users and 9,000 movies. We will use this in **Task 3**.
# * `ml-25m`: A massive dataset. We will extract a disjoint subset of movies from this to train our CV and NLP models in **Tasks 1 & 2**, ensuring our models learn on separate data before doing inference on the small graph.

# In[294]:


try:
    import google.colab
    IN_COLAB = True
except ImportError:
    import os
    print(os.getcwd())
    IN_COLAB = False


if IN_COLAB:
    import os
    os.makedirs('datasets', exist_ok=True)

    # Download and unzip the datasets
    get_ipython().system('wget -q https://files.grouplens.org/datasets/movielens/ml-latest-small.zip')
    get_ipython().system('wget -q https://files.grouplens.org/datasets/movielens/ml-25m.zip')
    get_ipython().system('unzip -q -o ml-latest-small.zip -d datasets/')
    get_ipython().system('unzip -q -o ml-25m.zip -d datasets/')


# ---
# ### **Stratified Sampling**
# To ensure our CV and NLP models generalize well to the final `ml-latest-small` graph, we perform **stratified sampling**. We calculate the distribution of "Decade + Primary Genre" in the small dataset, and force our training samples to match that exact distribution. If we train our CV and NLP models on a random subset of 15,000 movies from the massive 25M dataset, we might end up with a skewed distribution (e.g., too many 2010s Action movies and not enough 1950s Dramas) compared to our target ml-latest-small graph.

# In[295]:


import pandas as pd
import numpy as np

def extract_metadata(df):
    """Extracts Decade and Primary Genre to create a sampling stratum."""
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    df['decade'] = (df['year'] // 10 * 10).fillna(0).astype(int)
    df['primary_genre'] = df['genres'].str.split('|').str[0]
    df['stratum'] = df['decade'].astype(str) + "_" + df['primary_genre']
    return df


# In[296]:


print("Loading datasets...")
small_movies = pd.read_csv('datasets/ml-latest-small/movies.csv')
small_links = pd.read_csv('datasets/ml-latest-small/links.csv')

large_movies = pd.read_csv('datasets/ml-25m/movies.csv')
large_links = pd.read_csv('datasets/ml-25m/links.csv')

print("Processing metadata for stratified sampling...")
small_movies = extract_metadata(small_movies)
large_movies = extract_metadata(large_movies)

# Calculate target distribution from the small graph
target_distribution = small_movies['stratum'].value_counts(normalize=True)


# In[297]:


print(len(small_movies))
print(len(large_movies))
print(target_distribution)


# We prevent data leakage by creating a **disjoint** set of 15,000 movies to use for training on the image/text modalities.

# In[298]:


# Find disjoint set (Movies in 25M that are NOT in latest-small)
small_ids = set(small_movies['movieId'])
disjoint_movies = large_movies[~large_movies['movieId'].isin(small_ids)].copy()

print(f"Disjoint movies count: {len(disjoint_movies)}")

# Merge with TMDB links
# Discard movies with missing TMDB ids (database for plot summaries/posters)
disjoint_full = pd.merge(disjoint_movies, large_links, on='movieId')
disjoint_full = disjoint_full.dropna(subset=['tmdbId'])
disjoint_full['tmdbId'] = disjoint_full['tmdbId'].astype(int)

# Map weights and sample
stratum_counts = disjoint_full['stratum'].value_counts()
disjoint_full['sample_weight'] = disjoint_full['stratum'].map(
    lambda s: target_distribution.get(s, 0) / stratum_counts[s]
).fillna(0)

# Allows duplicates to better match the target distribution, especially for rare strata
# Major distributions are not affected much, while rare ones get more representation
# TODO: AVOID DUPLICATES AS MUCH AS POSSIBLE, ONLY WHEN EXHAUSTED
train_subset = disjoint_full.sample(n=30000, weights='sample_weight', replace=True, random_state=42)

# def sample_stratum(g):
#     target_n = int(target_distribution.get(g.name, 0) * 30000)
#     if target_n == 0:
#         return g.iloc[0:0]
#     if len(g) >= target_n:
#         return g.sample(n=target_n, random_state=42)  # no duplicates
#     else:
#         return g.sample(n=target_n, replace=True, random_state=42)  # duplicates only when exhausted

# train_subset = disjoint_full.groupby('stratum', group_keys=False).apply(sample_stratum)

# This is the skeleton's code for sampling: IT LEADS TO A COMPLETELY DIFFERENT TRAIN_DISTRIBUTION
# disjoint_full['sample_weight'] = disjoint_full['stratum'].map(target_distribution).fillna(0)
# train_subset = disjoint_full.sample(n=15000, weights='sample_weight', random_state=42)

train_distribution = train_subset['stratum'].value_counts(normalize=True)
print(train_distribution)

# Merge small_movies with their TMDB links as well
small_full = pd.merge(small_movies, small_links, on='movieId')
small_full = small_full.dropna(subset=['tmdbId'])
small_full['tmdbId'] = small_full['tmdbId'].astype(int)

columns_to_keep = ['movieId', 'title', 'genres', 'tmdbId']
train_subset = train_subset[columns_to_keep]
small_full = small_full[columns_to_keep]


# In[299]:


print(f"Stratified Training Subset: {len(train_subset)} movies.")
print(f"ML-Small Inference Target: {len(small_full)} movies.")


# ---
# ### **Fetch Multimodal Data (Posters & Plots)**
# We use the [TMDB API](https://developer.themoviedb.org/docs/getting-started) to asynchronously download posters and summaries. We fetch data for **both** the training set and the ML-Small set (needed for GNN embeddings later).
# 
# In order to access the API, you will need to sign up and generate an API key.

# In[300]:


import asyncio
import aiohttp
from tqdm.asyncio import tqdm
from getpass import getpass

NO_DATA = False

if NO_DATA:
    TMDB_API_KEY = getpass("Enter TMDB API Key: ")
IMAGE_DIR = "datasets/posters/"
os.makedirs(IMAGE_DIR, exist_ok=True)
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w185"

async def fetch_movie_data(session, tmdb_id, semaphore):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    async with semaphore:
        try:
            async with session.get(url) as response:
                if response.status != 200: return None
                data = await response.json()
                overview = data.get('overview', '')
                poster_path = data.get('poster_path', '')

                if not poster_path or not overview: return None

                img_url = f"{IMAGE_BASE_URL}{poster_path}"
                async with session.get(img_url) as img_response:
                    if img_response.status == 200:
                        img_filename = f"{tmdb_id}.jpg"
                        with open(os.path.join(IMAGE_DIR, img_filename), 'wb') as f:
                            f.write(await img_response.read())
                        return {"tmdbId": tmdb_id, "overview": overview, "poster_file": img_filename}
        except:
            return None

async def download_pipeline(df, output_csv_name):
    tmdb_ids = df['tmdbId'].unique()
    semaphore = asyncio.Semaphore(30)
    connector = aiohttp.TCPConnector(limit_per_host=30)

    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"\nFetching data for {output_csv_name}...")
        tasks = [fetch_movie_data(session, tid, semaphore) for tid in tmdb_ids]
        results = await tqdm.gather(*tasks)

    valid_results = [res for res in results if res is not None]
    text_df = pd.DataFrame(valid_results)
    final_df = pd.merge(df, text_df, on='tmdbId')
    final_df.to_csv(f'datasets/{output_csv_name}', index=False)


# This takes about 10 minutes.

# In[301]:


if NO_DATA:
    await download_pipeline(train_subset, "train_metadata_full.csv")
    await download_pipeline(small_full, "ml_small_metadata_full.csv")
    print("\nData preparation complete!")
    get_ipython().system('zip -r datasets.zip datasets/')
else:
    # We use the already downloaded csvs 
    # train_subset.to_csv("train_metadata_full.csv", index=False)
    # small_full.to_csv("ml_small_metadata_full.csv", index=False)
    print("\nData preparation complete!")


# ## **Task 1:  Multi-Label Genre Classification of Movie Posters**
# 
# The first task of the assignment involves training a convolutional model to predict movie genres based on poster images and extracting the produced visual features.

# ### **The PyTorch Dataset**
# Let's create a PyTorch Custom Dataset for the CV portion.

# In[302]:


if IN_COLAB: 
    from google.colab import drive
    drive.mount('/content/drive')
    # Assuming datasets.zip is in the root of your Google Drive
    get_ipython().system('unzip -o /content/drive/MyDrive/datasets.zip -d datasets/')


# In[303]:


import pandas as pd

# Load train_metadata_full.csv into train_subset
train_subset = pd.read_csv('./datasets/train_metadata_full.csv')
small_full = pd.read_csv('./datasets/ml_small_metadata_full.csv')

print(len(train_subset))
print(len(small_full))

print("train_metadata_full.csv loaded successfully into train_subset.")
print("ml_small_metadata_full.csv loaded successfully into small_full.")


# **Multi-Hot Encoding the Target Variables**
# 
# You will frame genre prediction as a **Multi-Label Classification** problem. The required loss function expects the target to be a binary matrix (0s and 1s).
# 
# However, the genres look like this:

# In[304]:


train_subset['genres'].head()


# Use the correct binarizer from scikit-learn to transform the pipe-separated genres into the correct format.

# In[305]:


from sklearn.preprocessing import MultiLabelBinarizer

# TODO: Instantiate the correct binarizer from sklearn for multi-label classification
mlb = MultiLabelBinarizer()

#TODO: complete encode_genres
def encode_genres(df):
    """
    Performs multi-hot encoding on pipe-delimited genre strings.

    Args:
        df (pd.DataFrame): Input data containing a 'genres' column.

    Returns:
        pd.DataFrame: The input DataFrame expanded with binary columns
                      for each unique genre identified by the binarizer.
    """

    # 1. Preprocess the 'genres' column so it is in the correct format for the binarizer.
    genres = [str(g).split("|") if pd.notna(g) else [] for g in df["genres"]]
    
    # 2. Generate the multi-hot encoded matrix using your binarizer.
    multi_hot = mlb.fit_transform(genres)    

    # 3. Return the new expanded dataframe.
    genres_df = pd.DataFrame(multi_hot, columns=mlb.classes_, index=df.index)
    return pd.concat([df, genres_df], axis=1)

train_subset = encode_genres(train_subset)
small_full = encode_genres(small_full)

print(f"Total unique genres: {len(mlb.classes_)}")
display(train_subset.head(3))


# Notice that TMDB posters are rectangular (~185x278).

# In[306]:


import os
from PIL import Image
import matplotlib.pyplot as plt

id = train_subset.iloc[42]['tmdbId']

# Define the image directory
IMAGE_DIR = "./datasets/posters/"

# Construct the image path
image_filename = f"{id}.jpg"
image_path = os.path.join(IMAGE_DIR, image_filename)

image = Image.open(image_path)

# Display the image
plt.imshow(image)
plt.title(f"Original Movie Poster")
plt.axis('off')
plt.show()

# Print its dimensions
print(f"Image dimensions: {image.size[1]}x{image.size[0]} pixels (Height x Width)")


# **Exploratory Data Analysis**
# 
# Before building any classification model, it is critical to understand the distribution of your target labels. If 80% of our movies are Dramas and only 2% are Westerns, the model might simply learn to predict "Drama" every time and ignore "Western" completely.
# 
# We will deal with this later.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

genre_columns = mlb.classes_

# Sum the occurrences of each genre
genre_counts = train_subset[genre_columns].sum().sort_values(ascending=False)

# Plotting the distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette="viridis", legend=False)
plt.title("Genre Distribution in Training Dataset (Label Imbalance)", fontsize=14)
plt.xlabel("Number of Movies", fontsize=12)
plt.ylabel("Genre", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Print the top 3 and bottom 3 genres
print("Top 3 Most Frequent Genres:")
print(genre_counts.head(3))
print("\nBottom 3 Least Frequent Genres:")
print(genre_counts.tail(3))

# IMAX and (no genres listed) are not really identifiable genres and have very few samples...
drop_classes = ['(no genres listed)', 'IMAX']
genre_columns_train = [g for g in genre_columns if g not in drop_classes]
num_classes = len(genre_columns_train)

# Drop movies with ONLY dropped labels (would have zero-label rows now)
keep_mask = train_subset[genre_columns_train].sum(axis=1) > 0
train_subset = train_subset[keep_mask].reset_index(drop=True)


# CNNs (like ResNet) expect square inputs, so you **must** apply a transform to resize/crop them.
# Since we are training from scratch, we will use a resolution of **128x128 pixels** to ensure the model trains reasonably fast and fits within memory constraints.
# 
# Let's package the posters into a ``MoviePosterDataset`` class.

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MoviePosterDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Get the label columns directly from mlb.classes_
        # This ensures we always select only the genre columns by their names
        self_label_cols_ = list(mlb.classes_)
        self.label_cols = [col for col in self.dataframe.columns if col in self_label_cols_]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # TODO: Retrieve the image and its corresponding multi-hot labels for the given `idx`.
        # Make sure the image is opened as an RGB PIL image, processed through
        # `self.transform`, and that the labels are cast to a `torch.float32` tensor.
        # Return image, labels.

        tmdbid = int(self.dataframe.iloc[idx]["tmdbId"])
        image_path = os.path.join(self.image_dir, f"{tmdbid}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(self.dataframe.iloc[idx][self.label_cols].values.astype(float), dtype=torch.float32)
        
        return image, labels


target_size = 128

# TODO: Define transforms (Resize, CenterCrop, ToTensor, Normalize)
test_transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize((target_size, target_size)),
    # Augmentation — training only
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(5),
    # transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # transforms.RandomErasing(p=0.3)  # randomly masks patches of the image
])


# In[309]:


from sklearn.model_selection import train_test_split

# Split the Training Data into Train and Validation (80/20)
train_df, val_df = train_test_split(train_subset, test_size=0.2, random_state=42)

# Instantiate the Datasets
print("Initializing Datasets...")
train_dataset = MoviePosterDataset(dataframe=train_df, image_dir=IMAGE_DIR, transform=train_transform)
val_dataset = MoviePosterDataset(dataframe=val_df, image_dir=IMAGE_DIR, transform=test_transform)

# Use the ML-Small dataset as our Test Set!
test_dataset = MoviePosterDataset(dataframe=small_full, image_dir=IMAGE_DIR, transform=test_transform)


# You can save `train_df, val_df` to use in following parts without having to regerate them, using the pickle format.
# 
# 
# 

# In[310]:


import pickle as pkl
pkl.dump(train_df, open("train_df.pkl", "wb"))
pkl.dump(val_df, open("val_df.pkl", "wb"))


# In[311]:


import numpy as np

image, labels = train_dataset[42]
# Convert tensor to numpy array and move channels to the last dimension (H, W, C)
image = image.numpy().transpose((1, 2, 0))
# # Clip values to [0, 1] as some values might be slightly outside due to floating point operations
image = np.clip(image, 0, 1)

# Get the genre names corresponding to the '1's in the labels
# The label_cols from MoviePosterDataset stores the order of genres
label_cols = train_dataset.label_cols
active_genres = [genre for i, genre in enumerate(label_cols) if labels[i] == 1]
genres_str = ', '.join(active_genres) if active_genres else 'No genres listed'

# Display the image
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(f"Processed Movie Poster\nGenres: {genres_str}")
plt.axis('off')
plt.show()


# In[312]:


batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)


print(f"Train Size: {len(train_dataset)} | Val Size: {len(val_dataset)} | Test (ML-Small) Size: {len(test_dataset)}")

print("For Training: ")
print(f"Number of Batches per Epoch: {len(train_loader)}")

sample_images, sample_labels = next(iter(train_loader))
print(f"Image Batch Shape: {sample_images.shape} -> [Batch, Channels, Height, Width]")
print(f"Label Batch Shape: {sample_labels.shape} -> [Batch, Num_Genres]")


# ---
# ### **Model Architecture (Multi-Label Classification)**
# You will now build a Custom CNN from scratch.
# 
# Predicting genres is a **Multi-Label Classification** problem. A movie is rarely just one genre; it can be an *Action*, *Sci-Fi*, AND *Thriller* simultaneously.
# 
# **Your Task:**
# 1. Design a CNN feature extractor using `nn.Conv2d`, `nn.ReLU`, and `nn.MaxPool2d`. A typical architecture stacks 3 to 4 of these blocks.
# 2. Flatten the spatial dimensions and pass the features through a fully connected (`nn.Linear`) classification head.
# 3. The final output dimension must exactly match the number of unique genres in our dataset.
# 4. *Do not* apply a Softmax or Sigmoid layer at the very end of your `forward` pass. Why? Because PyTorch's recommended multi-label loss function expects raw, unnormalized logits.

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # CxHxW = 3x128x128
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool2d(2)

        # CxHxW = 32x64x64
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=8, num_channels=64)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool2d(2)
        
        # CxHxW = 64x32x32
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn3   = nn.GroupNorm(num_groups=8, num_channels=128)
        self.relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool3 = nn.MaxPool2d(2)
        
        # CxHxW = 128x16x16
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn4   = nn.GroupNorm(num_groups=8, num_channels=256)
        self.relu4 = nn.LeakyReLU(negative_slope=0.01)
        self.pool4 = nn.MaxPool2d(2)
        
        # CxHxW = 256x8x8
        self.gap  = nn.AdaptiveAvgPool2d(1)  # 256x1x1

        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(128, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.extract_embeddings(x)
        x = self.relu_fc1(self.fc1(x))
        return self.fc2(self.drop(x))

    def extract_embeddings(self, x):
        """
        Helper function for Task 3:
        Returns the 256-dim visual embedding before the classifier.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.gn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.gn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.gn4(self.conv4(x))))
        x = self.gap(x).view(x.size(0), -1)
        return x


# In[ ]:


# Initialize the model
num_genres = len(mlb.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cv_model = CustomCNN(num_classes=num_genres).to(device)
print(f"Model initialized on: {device}")

if True:
    cv_model.load_state_dict(torch.load('best_model.pt'))
    print("Model weights loaded from best_model.pt")


# ---
# ### **Training Loop & Loss Function**
# To train this model, you need a loss function capable of handling multiple correct labels per image.
# Standard `CrossEntropyLoss` (used for multi-class problems like identifying cats vs. dogs) will **not** work here, as it assumes only one class can be true.
# 
# **Your Task:**
# 1. Define the correct loss function.
# 2. Define an optimizer (e.g., `Adam` or `AdamW`) with a learning rate of your choice (e.g., `1e-3` or `1e-4`).
# 3. Complete the forward and backward pass inside the training loop.

# Our dataset is heavily imbalanced. If we train normally, the model will just learn to predict "Drama" and "Comedy" for everything.
# 
# To fix this, we will calculate the **Positive Weight (`pos_weight`)** for each genre. This tells the loss function to assign a higher penalty when the model makes a mistake on a rare genre.
# The formula is: `pos_weight = (Total Samples - Positive Samples) / Positive Samples`

# In[ ]:


# Get the number of positive occurrences for each genre in the training set
pos_counts = train_df[genre_columns_train].sum().values
total_samples = len(train_df)

# Calculate pos_weight: (negatives / positives)
pos_weights = (total_samples - pos_counts) / pos_counts
# Take the square root to dampen extreme values
smoothed_weights = np.sqrt(pos_weights)
smoothed_weights = np.clip(smoothed_weights, 1.0, 15.0)

# Convert to a PyTorch tensor and move to the device (GPU/CPU)
pos_weight_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

print("Calculated pos_weight for the first 5 genres:")
for i in range(5):
    print(f"{genre_columns_train[i]}: {pos_weight_tensor[i].item():.2f}")


# In[ ]:


def healthcheck(model, loader, device, criterion):
    """
    Runs one forward+backward pass on a single batch and reports:
      - Per-activation-layer: fraction of outputs <=0 overall, and count of
        "dead" channels (>95% of their activations in the <=0 region). Works
        for ReLU / LeakyReLU / SiLU / GELU — detects effective channel death
        even when the activation never outputs literal zero.
      - Per-parameter: #params, ||w||, ||g||, g.std(), #zero_g, %zero_g.
      - Overall tallies: total params, total zero-grad params,
        total ||w||, total ||g||, weight penalty.
    """
    model.eval()
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device).float()

    # --- Per-channel dead activation check ---
    dead_stats = {}
    hooks = []
    def make_hook(name):
        def _hook(_, __, out):
            neg_mask = (out <= 0).float()
            overall_neg = neg_mask.mean().item()
            # Per-channel negative fraction: reduce over batch + spatial dims
            if out.ndim == 4:        # [B, C, H, W]
                per_ch_neg = neg_mask.mean(dim=(0, 2, 3))
            elif out.ndim == 2:      # [B, C]
                per_ch_neg = neg_mask.mean(dim=0)
            else:
                per_ch_neg = neg_mask.flatten()
            n_ch = per_ch_neg.numel()
            dead_ch = (per_ch_neg > 0.95).sum().item()
            dead_stats[name] = (overall_neg, dead_ch, n_ch)
        return _hook
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.SiLU, nn.GELU)):
            hooks.append(m.register_forward_hook(make_hook(name)))

    model.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    for h in hooks: h.remove()

    print("=== Activation health (<=0 fraction, dead channels) ===")
    for n, (neg_frac, dead_ch, n_ch) in dead_stats.items():
        flag = "  DEAD" if dead_ch > 0.5 * n_ch else ""
        print(f"  {n:15s} neg={neg_frac:.1%}  dead_ch={dead_ch:>3d}/{n_ch}{flag}")

    # --- Weight + gradient norms + param tally ---
    print("\n=== Weights & Gradients per layer ===")
    print(f"  {'layer':25s} {'#params':>9s} {'|w|':>10s} {'|g|':>10s} "
          f"{'g_std':>10s} {'#zero_g':>9s} {'%zero_g':>8s}")
    total_w_sq, total_g_sq = 0.0, 0.0
    total_params, total_zero_g = 0, 0
    for name, p in model.named_parameters():
        if p.grad is None: continue
        n_params = p.numel()
        w_norm = p.data.norm().item()
        g_norm = p.grad.norm().item()
        n_zero_g = int((p.grad.abs() < 1e-8).sum().item())
        pct_zero = n_zero_g / n_params
        total_w_sq += w_norm ** 2
        total_g_sq += g_norm ** 2
        total_params += n_params
        total_zero_g += n_zero_g
        print(f"  {name:25s} {n_params:>9d} {w_norm:10.2e} {g_norm:10.2e} "
              f"{p.grad.std().item():10.2e} {n_zero_g:>9d} {pct_zero:>7.1%}")

    total_w = total_w_sq ** 0.5
    total_g = total_g_sq ** 0.5
    overall_zero_pct = total_zero_g / max(total_params, 1)
    print(f"\n  TOTAL params:     {total_params:,} | zero-grad: {total_zero_g:,} ({overall_zero_pct:.1%})")
    print(f"  TOTAL weight norm:   {total_w:.4f}")
    print(f"  TOTAL gradient norm: {total_g:.4f}")
    print(f"  Weight penalty (wd * ||w||^2 / 2, wd=1e-4): {1e-4 * total_w_sq / 2:.6f}")
    return total_w, total_g


# In[ ]:


import torch.optim as optim
from tqdm.notebook import tqdm

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

def run_training(model, optimizer, scheduler, epochs):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"batch_loss": loss.item()})

        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_2.pt")

        print(f"End of Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch % 10 == 9:
            healthcheck(cv_model, train_loader, device, criterion)

    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(f"best_model_2.pt"))
    return train_losses, val_losses

optimizer = optim.Adam(filter(lambda p: p.requires_grad, cv_model.parameters()), lr=5e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)
train_losses, val_losses = run_training(cv_model, optimizer, scheduler, epochs=150)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

total_epochs  = len(train_losses)
xs = range(1, total_epochs + 1)

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(xs, train_losses, label="Train Loss", color="steelblue")
ax.plot(xs, val_losses,   label="Val Loss",   color="darkorange")
ax.set_title("Training & Validation Loss — All Stages", fontsize=14)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


# ---
# ### **Model Evaluation & Qualitative Analysis**
# 
# Now that your Custom CNN is trained, we need to evaluate how well it actually learned to identify genres.
# 
# In standard multi-class classification (e.g., identifying a dog vs. a cat), we usually just look at "Accuracy." However, in **multi-label classification**, exact-match accuracy is often overly strict and misleading. For example, if a movie is truly "Action|Sci-Fi|Thriller" and your model predicts "Action|Sci-Fi", it is partially correct, but exact accuracy would score it as a 0.
# 
# To properly evaluate your model, you will need to calculate metrics like **F1-Score** and **ROC-AUC**, and perform some qualitative "eye tests."
# 
# 
# 

# **1. Quantitative Metrics**
# Write an evaluation loop (using a separate validation/test set) to calculate the following:
# * **Sigmoid Activation:** Pass your model's raw output logits through a `torch.sigmoid()` function to convert them into probabilities (0.0 to 1.0).
# * **Thresholding:** Convert those probabilities into binary predictions (0 or 1) using a threshold (e.g., `0.5`).
# * **Macro F1-Score:** Calculate the F1-score for each genre independently, then take the unweighted average.
# * **Micro F1-Score:** Calculate the F1-score globally by counting the total true positives, false negatives, and false positives across all classes.

# In[ ]:


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

# ==========================================
# 1. Quantitative Evaluation
# ==========================================
cv_model.eval()

# TODO: Populate these lists with numpy arrays during your evaluation loop
all_targets = []
all_predictions = []

print("Evaluating model on the Test Set (ML-Small)...")

# TODO:
# 1. Write the evaluation loop over `test_loader` (remember to disable gradients)
# 2. Apply the correct activation function to get probabilities.
# 3. Append the true labels to `all_targets` and probabilities to `all_predictions`.
# 4. Stack them into continuous 2D numpy arrays.
# 5. Create a `binary_predictions` array by thresholding probabilities at 0.5.

# --- YOUR EVALUATION LOOP HERE ---

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cv_model(images)
        probs = torch.sigmoid(outputs)
        all_targets.append(labels.cpu().numpy())
        all_predictions.append(probs.cpu().numpy())

all_targets = np.vstack(all_targets)
all_predictions = np.vstack(all_predictions)
binary_predictions = (all_predictions >= 0.3).astype(int)

# TODO: Calculate the metrics.
# Hint: Pay attention to which metric needs binary predictions vs. continuous probabilities.
macro_f1 = f1_score(all_targets, binary_predictions, average='macro', zero_division=0)
micro_f1 = f1_score(all_targets, binary_predictions, average='micro', zero_division=0)
roc_auc  = roc_auc_score(all_targets, all_predictions, average='macro')

# ==========================================
# --- VISUALIZE QUANTITATIVE METRICS ---
# ==========================================
metrics_dict = {'Macro F1': macro_f1, 'Micro F1': micro_f1, 'ROC-AUC': roc_auc}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict.keys(), metrics_dict.values(), color=['#87CEEB', '#98FB98', '#FA8072'])
plt.ylim(0, 1.1)
plt.title('Test Set Evaluation Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 
# **2. Qualitative Analysis**
# Visualize your model's predictions to see if they make logical sense.
# * Write a function to display a grid of 5 random movie posters from your dataset.
# * Above each poster, print the **True Genres** and the **Predicted Genres** (any genre where the predicted probability was > 0.5).

# In[ ]:


# ==========================================
# 2. Qualitative  (Visual Grid)
# ==========================================
def imshow_denormalized(img):
    img = img / 2 + 0.5
    img = np.clip(img, 0, 1)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def qual(cv_model):
    # TODO: Extract exactly ONE batch of images and labels from the test_loader
    images, labels = next(iter(test_loader))

    # TODO: Run the images through the model (without gradients) to get binary predictions (0 or 1)
    # Note: You can reuse your threshold = 0.5
    with torch.no_grad():
        outputs = cv_model(images.to(device))
        probs = torch.sigmoid(outputs).cpu()
    preds = (probs >= 0.5).int()


    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    genre_names = mlb.classes_

    for i in range(20):
        ax = axes[i // 5][i % 5]

        # TODO: For the i-th image in the batch, extract the list of actual text strings
        # for both the true genres and the predicted genres using `genre_names`.
        # Hint: You will need to find which indices contain a '1' in the labels and preds arrays.
        true_genres = [genre_names[j] for j in range(len(genre_names)) if labels[i][j] == 1]
        pred_genres = [genre_names[j] for j in range(len(genre_names)) if preds[i][j] == 1]

        # ==========================================
        # --- VISUALIZATION ---
        # ==========================================
        plt.sca(ax)
        imshow_denormalized(images[i])
        ax.axis('off')

        title_text = f"True: {', '.join(true_genres)}\nPred: {', '.join(pred_genres) if pred_genres else 'None'}"
        color = 'green' if set(true_genres) == set(pred_genres) else 'black'
        ax.set_title(title_text, fontsize=10, wrap=True, color=color)

    plt.tight_layout()
    plt.show()

qual(cv_model)


# Answer below:
# 
# 1. Why is standard "exact-match accuracy" a poor metric for this specific task? What happens to your F1-Score if you predict *every* movie is a "Drama" and "Comedy"?
# 2. **Threshold Tuning:** In your evaluation, you likely used a probability threshold of 0.5 to decide if a genre was present. If you lower this threshold to 0.3, how would you expect your Precision and Recall to change? Explain why.
# 3. **Class Imbalance:** Look at your per-class F1-scores. Which genres did your model perform best on, and which did it struggle with? Does this correlate with the frequency of those genres in the training data?
# 4. **Visual Ambiguity:** Based on your visual "eye test" of the posters, name one genre that you think is visually easy for a CNN to learn, and one genre that is visually ambiguous. Provide a brief justification based on how movie posters are designed.

# Your answers:
# 
# 

# ### **Optimization: Transfer Learning with Pre-trained ResNet**
# Instead of training a CNN from scratch, you will now leverage Transfer Learning. Load a pre-trained ResNet18 or ResNet50 model from torchvision.models. Modify its final classification layer to output the correct number of genres, fine-tune it on the training set, and perform the same quantitative evaluation (Macro F1, Micro F1, ROC-AUC) on the ML-Small test set.
# 
# Python

# In[ ]:


import torchvision.models as models
from sklearn.metrics import f1_score, roc_auc_score

# ==========================================
# Pretrained ResNet Initialization
# ==========================================

# 1. Load ResNet18 with ImageNet weights
resnet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)

num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.LeakyReLU(0.01, inplace=True),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

def run_resnet_stage(model, optimizer, epochs, stage_name):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"{stage_name} Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({"batch_loss": loss.item()})
        train_loss = running_loss / len(train_dataset)
        train_losses.append(train_loss)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss = val_loss / len(val_dataset)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_resnet_{stage_name}.pt")

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

    print(f"{stage_name} complete! Best val loss: {best_val_loss:.4f}")
    model.load_state_dict(torch.load(f"best_resnet_{stage_name}.pt"))
    return train_losses, val_losses

# ==========================================
# Stage 1: Freeze backbone, train fc head only
# ==========================================
print("Stage 1: Training classifier head (backbone frozen)...")
for name, param in resnet_model.named_parameters():
    param.requires_grad = name.startswith("fc")

optimizer_s1 = optim.Adam(resnet_model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
train_losses_res_s1, val_losses_res_s1 = run_resnet_stage(
    resnet_model, optimizer_s1, epochs=5, stage_name="resnet_s1")

# ==========================================
# Stage 2: Unfreeze all, fine-tune end-to-end
# ==========================================
print("Stage 2: End-to-end fine-tuning...")
for param in resnet_model.parameters():
    param.requires_grad = True

optimizer_s2 = optim.Adam([
    {"params": [p for n, p in resnet_model.named_parameters() if not n.startswith("fc")], "lr": 1e-5},
    {"params": resnet_model.fc.parameters(), "lr": 1e-4},
], weight_decay=1e-4)
train_losses_res_s2, val_losses_res_s2 = run_resnet_stage(
    resnet_model, optimizer_s2, epochs=5, stage_name="resnet_s2")

print("ResNet training complete!")

# ==========================================
# Loss curve — both stages
# ==========================================
s1_ep = len(train_losses_res_s1)
train_all_res = train_losses_res_s1 + train_losses_res_s2
val_all_res   = val_losses_res_s1   + val_losses_res_s2
xs = range(1, len(train_all_res) + 1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xs, train_all_res, label="Train Loss", color="steelblue")
ax.plot(xs, val_all_res,   label="Val Loss",   color="darkorange")
ax.axvspan(0.5, s1_ep + 0.5,              alpha=0.15, color="#e8f4fd", label="Stage 1 (head only)")
ax.axvspan(s1_ep + 0.5, len(train_all_res) + 0.5, alpha=0.15, color="#eafaf1", label="Stage 2 (end-to-end)")
ax.axvline(s1_ep + 0.5, color="gray", linestyle="--", linewidth=0.8)
ax.set_title("ResNet Fine-tuning Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ==========================================
# Quantitative Evaluation
# ==========================================
resnet_model.eval()
all_targets_res = []
all_predictions_res = []

print("Evaluating model on the Test Set (ML-Small)...")
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet_model(images)
        probs = torch.sigmoid(outputs)
        all_targets_res.append(labels.cpu().numpy())
        all_predictions_res.append(probs.cpu().numpy())

all_targets_res = np.vstack(all_targets_res)
all_predictions_res = np.vstack(all_predictions_res)
binary_predictions_res = (all_predictions_res >= 0.3).astype(int)

macro_f1_res = f1_score(all_targets_res, binary_predictions_res, average="macro", zero_division=0)
micro_f1_res = f1_score(all_targets_res, binary_predictions_res, average="micro", zero_division=0)
roc_auc_res  = roc_auc_score(all_targets_res, all_predictions_res, average="macro")

# ==========================================
# Visualise Quantitative Metrics
# ==========================================
metrics_dict_res = {"Macro F1": macro_f1_res, "Micro F1": micro_f1_res, "ROC-AUC": roc_auc_res}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict_res.keys(), metrics_dict_res.values(), color=["#87CEEB", "#98FB98", "#FA8072"])
plt.ylim(0, 1.1)
plt.title("ResNet Test Set Evaluation Metrics", fontsize=14)
plt.ylabel("Score", fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}",
             ha="center", va="bottom", fontweight="bold", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# In[ ]:


# ==========================================
# --- VISUALIZE QUANTITATIVE METRICS ---
# ==========================================

metrics_dict_res = {'Macro F1': macro_f1_res, 'Micro F1': micro_f1_res, 'ROC-AUC': roc_auc_res}

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics_dict_res.keys(), metrics_dict_res.values(), color=['#87CEEB', '#98FB98', '#FA8072'])
plt.ylim(0, 1.1)
plt.title('Test Set Evaluation Metrics', fontsize=14)
plt.ylabel('Score', fontsize=12)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

qual(resnet_model)


# Answer Below:
# 
# 1. **Performance Comparison:** Compare the Macro F1 and ROC-AUC scores of your Custom CNN against the Pre-trained ResNet. Which model performed better, and by how much?
# 
# 2. **Training Dynamics:** Look at the training and validation loss curves for both models. Did one model converge faster than the other? Did you notice any signs of overfitting in either model?
# 
# 3. Why do you think the Pre-trained ResNet achieves these results compared to your Custom CNN, considering both models were trained on the exact same movie poster dataset?

# ---
# ### **Extract Visual Embeddings**
# 
# 
# Now that you have trained both a Custom CNN and a Pre-trained ResNet to understand movie genres based on visual features, we want to extract that "visual understanding" to use in our recommendation system graph (Task 3).
# 
# **Look back at your evaluation metrics (F1 Score, ROC-AUC) and choose the model that achieved the BEST overall results.**
# 
# Instead of getting the final genre predictions (the raw logits), we will pass the posters from the  `ml-latest-small` dataset through your chosen model and stop at the **penultimate layer**. This gives us a dense feature vector (embedding) for every single movie. (Note: The size of this vector will depend on the model you chose—for example, a standard ResNet18 outputs a 512-dimensional vector).
# 
# Ensure the order of the embeddings perfectly matches the dataframe! Save the resulting tensor to a `.pt` (PyTorch) file so it can be easily loaded in Task 3.
# 

# In[ ]:


import torch
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

# TODO: 1. Choose your best model (e.g., cv_model or resnet_model)
best_vision_model =


# Setup DataLoader (CRITICAL: shuffle MUST be False to match embeddings to movieIds!)
inference_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Set the chosen model to evaluation mode
best_vision_model.eval()
all_embeddings = []

print(f"Extracting embeddings for {len(test_dataset)} movies...")

# TODO: 2. Write the extraction loop
# - Loop over the images in `inference_loader`.
# - Extract the corresponding embeddings.
# - Move the resulting embeddings back to `.cpu()` and append them to the `all_embeddings` list.

# --- YOUR EXTRACTION LOOP HERE ---





# ---------------------------------

# Concatenate all batches into a single large tensor
final_visual_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Extracted Embeddings Shape: {final_visual_embeddings.shape} -> [Num_Movies, Embedding_Dim]")

# ==========================================
# --- SAVING BOILERPLATE ---
# (Do not modify the code below this line)
# ==========================================
save_dict = {
    'movie_ids': torch.tensor(small_full['movieId'].values),
    'embeddings': final_visual_embeddings
}

torch.save(save_dict, 'datasets/cv_embeddings.pt')
print("Visual embeddings successfully saved to 'datasets/cv_embeddings.pt'!")


# ---
# ## **Task 2: Natural Language Processing for Genre Classification**
# 
# In this task, you will tackle the exact same multi-label classification problem, but using a completely different modality: **Text**. You will build a sequence model (RNN, LSTM, or GRU) to predict genres based on the movie's plot summary (the `overview` column).
# 
# To make this model more powerful and interpretable, you are required to implement a **Custom Attention Mechanism**.

# ### **Text Preprocessing & Data Loading**
# Unlike images, neural networks cannot process raw text. We must convert the words into numerical tokens. Below, we provide the pipeline to:
# 1. Clean and tokenize the text.
# 2. Build a Vocabulary based solely on the training set (to avoid data leakage).
# 3. Convert summaries into padded numerical tensors.

# In[ ]:


# Assuming datasets.zip is in the root of your Google Drive
get_ipython().system('unzip -o /content/drive/MyDrive/datasets.zip -d datasets/')


# In[ ]:


import pickle as pkl

# load pre-saved pickled dataframes, if needed
train_df = pkl.load(open('train_df.pkl', 'rb'))
val_df = pkl.load(open('val_df.pkl', 'rb'))


# A tokenizer is essentially the "translator" that turns a raw blob of text into a format a machine can actually digest.
# 
# Complete the simple tokenizer, as instructed below.

# In[ ]:


import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Simple Tokenizer, returns a list of strings
def tokenize(text):
    """Lowercases, removes punctuation, and splits by whitespace."""
    # TODO
    # 1. Convert the input text to a lowercase string.
    # 2. Use regex to remove any character that isn't a letter, number, or whitespace.
    # 3. Split the cleaned string into a list of individual words (tokens).

    # --- YOUR CODE HERE ---



    # ----------------------


# Build Vocabulary from Training Set ONLY
print("Building vocabulary...")
all_tokens = []
for plot in train_df['overview']:
    all_tokens.extend(tokenize(plot))


# We need to build a mapping of the words within the plot summaries. We select our vocabulary to contain the top 10,000 most frequent words and map them to unique IDs. It reserves 0 for padding and 1 for unknown words (the rare ones we excluded). Raw text needs to be converted into fixed-length numerical vectors (here, set to 150 tokens), truncating long sequences and padding short ones to ensure uniform input for the model.

# In[ ]:


# Keep top 10,000 most common words, reserve 0 for <PAD> and 1 for <UNK>

max_vocab_size = 10000
vocab_counts = Counter(all_tokens)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(vocab_counts.most_common(max_vocab_size))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def text_to_tensor(text, vocab, max_len=150):
    """Converts a text string to a padded tensor of token IDs."""
    tokens = tokenize(text)
    token_ids = [vocab.get(word, vocab['<UNK>']) for word in tokens]

    # Truncate if too long
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    # Pad if too short
    pad_len = max_len - len(token_ids)
    token_ids = token_ids + [vocab['<PAD>']] * pad_len

    return torch.tensor(token_ids, dtype=torch.long)


# And we will build a pytorch dataset like before.

# In[ ]:


class MoviePlotDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.dataframe = dataframe.reset_index(drop=True)
        self.vocab = vocab
        self.label_cols = self.dataframe.columns[6:].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        plot_text = self.dataframe.loc[idx, 'overview']
        text_tensor = text_to_tensor(plot_text, self.vocab)

        # Extract labels
        labels = self.dataframe.loc[idx, self.label_cols].values.astype(float)
        labels = torch.tensor(labels, dtype=torch.float32)

        return text_tensor, labels


# In[ ]:


print("Initializing NLP Datasets...")
nlp_train_dataset = MoviePlotDataset(train_df, vocab)
nlp_val_dataset = MoviePlotDataset(val_df, vocab)
nlp_test_dataset = MoviePlotDataset(small_full, vocab)


# In[ ]:


batch_size = 32
nlp_train_loader = DataLoader(nlp_train_dataset, batch_size=batch_size, shuffle=True)
nlp_val_loader = DataLoader(nlp_val_dataset, batch_size=batch_size, shuffle=False)
nlp_test_loader = DataLoader(nlp_test_dataset, batch_size=batch_size, shuffle=False)

# Test the dataloader
print("For training...")
sample_texts, sample_labels = next(iter(nlp_train_loader))
print(f"Text Batch Shape: {sample_texts.shape} -> [Batch, Sequence_Length]")
print(f"Label Batch Shape: {sample_labels.shape} -> [Batch, Num_Genres]")


# ---
# ### **Architecture with Custom Attention**
# 
# You must build a deep learning text classifier. You may choose the recurrent backbone (e.g., `nn.RNN`, `nn.GRU`, or `nn.LSTM`).
# 
# However, instead of simply passing the final hidden state to your classification head, you must implement a **Self-Attention Mechanism** to compute a weighted context vector from *all* the hidden states of your sequence.
# 
# **A simple Attention Mechanism:**
# Given the hidden states of your RNN layer $H = [h_1, h_2, ..., h_T]$:
# 1. Calculate the attention scores for each token: $e_t = \text{Linear}(h_t)$
# 2. Compute the normalized attention weights using Softmax: $\alpha = \text{softmax}(e)$
# 3. Compute the final context vector by taking the weighted sum of the hidden states: $c = \sum_{t=1}^{T} \alpha_t h_t$
# 

# 
# In the following code, you will need to:
# 1. Define an `nn.Embedding` layer.
# 2. Define your recurrent layer (`batch_first=True` is recommended).
# 3. Implement the attention mechanism in the `forward` pass.
# 4. Pass the resulting context vector $c$ to a fully connected layer to output your unnormalized logits.
# 5. Implement the `extract_embeddings` function to return the context vector $c$.

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class CustomTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(CustomTextClassifier, self).__init__()

        # TODO: Define your layers
        # 1. Embedding layer: map vocab_size to embed_dim (don't forget padding_idx=0)
        # 2. RNN layer: set batch_first=True
        # 3. Attention Projection: a Linear layer mapping hidden_dim to a single score (1)
        # 4. Classifier Head: a Linear layer mapping hidden_dim to num_classes

        # --- YOUR LAYERS HERE ---
        self.embedding =

        self.rnn =

        self.attention_proj =

        self.fc =
        # ------------------------


    def forward(self, x):

        # TODO: Implement the Attention-based Forward Pass
        # 1. Pass input through the embedding layer.
        # 2. Pass embeddings through the RNN. Get all hidden states 'H'.
        # 3. Calculate energy scores 'e'
        # 4. Calculate attention weights 'alpha'.
        # 5. Calculate the context vector 'c'.
        # 6. Pass the context vector through the final classifier to get logits.

        # --- YOUR FORWARD PASS HERE ---

        return logits
        # ------------------------------



    def extract_embeddings(self, x):
        """Returns the final context vector 'c' BEFORE the classification head."""
        # TODO: Replicate the forward pass logic above, but STOP after calculating
        # the context vector 'c'. Return 'c' as the embedding.

        # --- YOUR EXTRACTION HERE ---

        return c
        # ----------------------------


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model will be initialized in {device}.\n")

vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
num_classes = len(nlp_train_dataset.label_cols)

nlp_model = CustomTextClassifier(vocab_size, embed_dim, hidden_dim, num_classes).to(device)
print(nlp_model)


# ---
# ### **Training & Evaluation Loop**
# 
# You already know how to train and evaluate a PyTorch model!
# 
# 1. Write the training loop.
# 2. Remember to use the `pos_weight` tensor we calculated in Task 1 to handle the class imbalance.
# 3. Plot the Training and Validation loss curves.
# 4. Evaluate the model on ML-Small summaries using Macro/Micro F1-Scores.

# In[ ]:


import torch.optim as optim
from tqdm.notebook import tqdm

# --- 1. SETUP ---
# TODO: Initialize the loss criterion and optimizer (use pose_weight)
criterion = ...
optimizer = ...

epochs = 10
train_losses = []
val_losses = []

# --- 2. TRAINING & VALIDATION LOOP ---
for epoch in range(epochs):
    # --- TRAINING PHASE ---
    nlp_model.train()
    running_train_loss = 0.0

    # TODO: Implement the training loop
    # 1. Iterate over train_loader
    # 2. Forward pass, Calculate Loss, Backward pass, Optimizer step
    # Optional: Use gradient clipping to prevent expoding gradients (common in RNNs)


    # --- VALIDATION PHASE ---
    nlp_model.eval()
    running_val_loss = 0.0

    # TODO: Implement the validation loop (remember torch.no_grad())
    # 1. Iterate over val_loader
    # 2. Forward pass, Calculate Loss


    # Save average losses for plotting
    # train_losses.append(...)
    # val_losses.append(...)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")



# ==========================================
# --- VISUALIZATION LOSS CURVES ---
# (Do not modify the code below this line)
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o', color='blue')
plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='o', color='orange')
plt.title('NLP Model Loss Over Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss (Weighted BCE)', fontsize=12)
plt.xticks(range(1, epochs+1))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[ ]:


# --- 3. EVALUATION N TEST SET (ML-Small) ---
# TODO: Evaluate on the test set
# - Generate predictions for the test_loader
# - Apply Sigmoid and 0.5 threshold
# - Calculate Macro/Micro F1 using sklearn

print("\nEvaluating on Test Set...")

# --- YOUR CODE  HERE ---



# ----------------------

macro_f1 = ...
micro_f1 = ...

print(f"Test Macro F1-Score: {macro_f1:.4f}")
print(f"Test Micro F1-Score: {micro_f1:.4f}")


# ---
# ### **Extract NLP Embeddings for Task 3**
# 
# Just like in Task 1, we need to extract the semantic "understanding" of the plots to use as node features in our GNN and save them in `nlp_embeddings.pt`.
# 

# In[ ]:


# TODO: Extract semantic text embeddings using your trained `nlp_model`
# 1. Create a list named `all_nlp_embeddings` to store batch results.
# 2. Write a loop over `nlp_test_loader` without tracking gradients to extract the embeddings.
# 3. Move the resulting embeddings back to the CPU and append them to your list.
# 4. Combine all batches into a single tensor named `final_nlp_embeddings`.

# --- YOUR EXTRACTION LOOP HERE ---



# ---------------------------------


# ==========================================
# --- SAVING ---
# (Do not modify the code below this line)
# ==========================================
print(f"Extracted NLP Embeddings Shape: {final_nlp_embeddings.shape} -> [Num_Movies, Hidden_Dim]")

save_dict = {
    'movie_ids': torch.tensor(small_full['movieId'].values),
    'embeddings': final_nlp_embeddings
}

torch.save(save_dict, 'datasets/nlp_embeddings.pt')
print("Semantic NLP embeddings successfully saved to 'datasets/nlp_embeddings.pt'!")


# ---
# ### **Questions**
# 
# 1. **Modality Comparison:** Look at your final F1-scores. Did the NLP model (Plot Summaries) or the CV model (Movie Posters) perform better at predicting genres? Why do you think that modality was more successful?
# 2. **Attention Interpretability:** One of the main benefits of the Attention mechanism is interpretability. If you were to visualize the attention weights ($\alpha$) for the summary *"A lone cowboy rides into a dusty town to face the outlaw,"* which words do you expect would receive the highest attention weights when predicting the "Western" genre?
# 3. **Sequence Length:** We capped the `max_len` at 150 words and padded shorter sequences with 0s. How might processing very long padding sequences negatively affect an RNN's performance, and how does the Attention mechanism help mitigate this issue?

# ---
# ### Optimizations
# ---
# 
# Your initial F1-scores are likely low. Training an NLP model from scratch on only 15,000 short summaries is difficult, especially with severe class imbalance. If you want to extract higher-quality semantic embeddings for Task 3, try implementing **some** of these upgrades:
# 
# **1. Increased epochs, hyperparameter tuning**
# 
# **3. Bidirectional RNNs**: Learn from left to right and right to left.
# This gives your Attention mechanism much better context to work with.
# 
# **3. Pre-trained Word Embeddings (GloVe)**
# Instead of forcing your `nn.Embedding` layer to learn the meaning of words from scratch, you can load pre-trained GloVe weights `torchtext.vocab.GloVe` (Global Vectors for Word Representation). This gives your model an immediate, massive understanding of the English language.
# 
# **4. Dynamic Thresholding**
# 

# In[ ]:


#TODO


# ---
# ## **Task 3: Graph Neural Networks for Link Prediction**
# 
# 
# In this final task, you will build a recommendation system. You will construct a **Bipartite Heterogeneous Graph** consisting of `User` nodes and `Movie` nodes, connected by `rates` edges.
# 
# The goal is **Link Prediction**: predicting the probability that a specific User will rate a specific Movie.
# 
# Instead of writing the complex GNN layers from scratch, the architecture and training loop are provided for you. You must engineer the node features using the embeddings you extracted in Tasks 1 & 2, format the graph data, and perform an ablation study to explain *why* multimodal features improve recommendation systems.
# 
# 

# ---
# ### **Graph Construction**
# Graph frameworks require node IDs to be contiguous integers starting from $0$. MovieLens IDs (like `movieId = 5000`) will cause out-of-bounds errors if not mapped properly. The code below loads the ratings and creates mappings.

# In[ ]:


get_ipython().system('unzip -o /content/drive/MyDrive/datasets.zip -d datasets/')


# In[ ]:


import torch
import os

print("Installing PyTorch Geometric base...")
get_ipython().system('pip install -q torch-geometric')

print("Attempting to install optional C++ speedups (this may take a minute)...")
torch_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda.replace('.', '')
whl_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"

# We use > /dev/null to hide the scary red text if the wheels aren't compiled for Colab's newest PyTorch version yet.
# Our GNN will run perfectly fine on the base library!
os.system(f"pip install -q pyg_lib torch_scatter torch_sparse -f {whl_url} > /dev/null 2>&1")

print("PyTorch Geometric environment ready!")


# In[ ]:


import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# Load the ratings data from the small graph
ratings_df = pd.read_csv('datasets/datasets/ml-latest-small/ratings.csv')

# Only keep ratings for movies that exist in our valid small_full set (with TMDB links)
small_full = pd.read_csv('datasets/datasets/ml_small_metadata_full.csv')
valid_movie_ids = set(small_full['movieId'])
ratings_df = ratings_df[ratings_df['movieId'].isin(valid_movie_ids)]

# Create Contiguous IDs for Users and Movies
unique_users = ratings_df['userId'].unique()
unique_movies = small_full['movieId'].unique() # Use the full valid subset, not just rated ones

user_mapping = {userid: i for i, userid in enumerate(unique_users)}
movie_mapping = {movieid: i for i, movieid in enumerate(unique_movies)}

# Map the edges
src = [user_mapping[uid] for uid in ratings_df['userId']]
dst = [movie_mapping[mid] for mid in ratings_df['movieId']]
edge_index = torch.tensor([src, dst], dtype=torch.long)

print(f"Total Users: {len(user_mapping)}")
print(f"Total Movies: {len(movie_mapping)}")
print(f"Total Ratings (Edges): {edge_index.shape[1]}")


# ---
# ### **Feature Engineering & Initialization**
# 
# A GNN relies on its starting node features. Because we don't have metadata for the users (like age or gender), we will initialize user nodes with random learnable embeddings. For the movies, you will test different feature combinations.
# 
# 1. Load your saved `cv_embeddings.pt` and `nlp_embeddings.pt`.
# 2. Ensure the order of the embeddings matches the contiguous `movie_mapping` (hint: they should already match if you used `small_full` sequentially!).
# 3. Initialize the PyG `HeteroData` object.
# 4. Experiment with different movie feature initializations:
#    * **Baseline:** Dummy features (e.g., `torch.ones` or random noise).
#    * **Unimodal:** Only CV features, or only NLP features.
#    * **Multimodal Concatenation:** Concatenate CV and NLP features (`torch.cat`).
#    * **Pretrained:** We have provided a file of pre-trained and fine-tuned (on the train_df movie plots) DistilBERT embeddings.

# Load your saved `cv_embeddings.pt` and `nlp_embeddings.pt`.

# In[ ]:


cv_data = torch.load('cv_embeddings.pt')
cv_features = cv_data['embeddings'] # Shape: [Num_Movies, 512]

nlp_data = torch.load('nlp_embeddings.pt')
nlp_features = nlp_data['embeddings'] # Shape: [Num_Movies, 256]


# Ensure the order of the embeddings matches the contiguous `movie_mapping`. They should already match if you used `small_full` sequentially
# 
# You can just use the provided embeddings to complete this task, if you could not complete the previous tasks successfully or the embeddings created were not great representations of the data (this will not affect your grade).
# 
# **Important:** make sure to normalize the extracted features! CV and NLP embeddings are raw outputs from custom networks and their range will blow up the gradients.

# In[ ]:


import torch.nn.functional as F

def get_movie_features(strategy, cv_feats, nlp_feats, num_movies):
    """Returns the appropriate node feature tensor based on the selected ablation strategy."""
    if strategy == "baseline":
        return torch.randn((num_movies, 64))

    elif strategy == "cv":
         # TODO: Return the L2-normalized visual embeddings from Task 1

    elif strategy == "nlp":
         # TODO: Return the L2-normalized visual embeddings from Task 1

    elif strategy == "multimodal":
         # TODO: Concatenate the CV and NLP features,
         # then apply L2-normalization to the combined vector.

    elif strategy == "pretrained":
         # TODO: Load the 'pretrained_nlp_embeddings.pt' file and return the normalized embeddings.

    else:
        raise ValueError("Invalid strategy! Please choose 'baseline', 'cv', 'nlp', 'multimodal', or 'pretrained'.")


# This function transforms raw data into a **Bipartite Heterogeneous Graph**. In high-level terms, it sets up the "map" that the GNN will use to navigate and learn.
# 
# * **`HeteroData()`**: Instead of a simple list, we initialize a specialized container that recognizes different "species" of nodes (Users vs. Movies) and their specific relationships.
# * **User Nodes (Structural IDs)**: We tell the graph how many users exist. Since we have no metadata for them, the GNN will later assign them unique, learnable "fingerprints" (embeddings) based purely on their position in the graph.
# * **Movie Nodes (Semantic Features)**: Unlike users, movies are initialized with the rich features you extracted (CV/NLP). This allows the GNN to understand *what* a movie is, rather than just knowing its ID.
# * **Edge Index (The Connections)**: This maps the "rates" relationship. It acts as the bridge that tells the model which users interacted with which movies.
# * **`T.ToUndirected()` (Two-Way Communication)**: By default, a rating is one-way (User $\to$ Movie). This transform creates "reverse" edges so that information can flow from movies back to users, allowing the GNN to perform complex message passing.

# In[ ]:


from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def build_graph(movie_features):
    data = HeteroData()

    # User nodes just need to know how many exist
    data['user'].num_nodes = len(user_mapping)

    # Movie nodes require the actual semantic features
    data['movie'].x = movie_features

    # Add Edges
    data['user', 'rates', 'movie'].edge_index = edge_index

    # GNNs need undirected paths to pass messages back and forth
    data = T.ToUndirected()(data)

    return data


# ---
# ### **GNN Architecture and Training**
# 
# Below is a complete implementation of a Heterogeneous Graph Neural Network using `GraphSAGE` ([paper](https://arxiv.org/abs/1706.02216)).
# 
# Imagine you are trying to represent a specific movie, let's say *The Matrix*.
# 1. **Initial State:** You start with the movie's own features (the CV poster embedding and the NLP plot embedding you gave it).
# 2. **Sampling:** GraphSAGE looks at the graph and randomly *samples* a few neighboring nodes. For a movie, its neighbors are the **Users** who rated it.
# 3. **Aggregating** It asks those neighboring Users, *"What else do you like?"* It pulls in the features of the *other* movies those users watched, mathematically averages (or pools) them together, and uses that new information to update the embedding for *The Matrix*.
# 
# By layer 2 of your GNN, the embedding for a User is no longer just random noise. It has been updated to represent a combination of all the CV posters and NLP plots of the movies they have watched! The GNN learns to say: *"User A tends to click on movies with dark, blue-tinted posters (CV) and plots involving space travel (NLP)."* When the Decoder compares User A to a new sci-fi movie, the dot product of their embeddings will be very high, resulting in a successful recommendation!
# 
# Specifically:
# * **The Encoder:** Two layers of message passing. It aggregates features from neighboring nodes to create rich, relational embeddings for both users and movies.
# * **The Decoder (Link Predictor):** To predict if an edge exists, we take the final GNN embedding of a User ($z_u$) and a Movie ($z_v$), and compute their dot product: $\hat{y} = z_u \cdot z_v$. We then pass this through a Sigmoid to get a probability.

# In[ ]:


import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv

class HeteroGNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # HeteroConv allows us to define specific message passing for specific edge types.
        self.conv1 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), hidden_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user', 'rates', 'movie'): SAGEConv((-1, -1), out_channels),
            ('movie', 'rev_rates', 'user'): SAGEConv((-1, -1), out_channels),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict


class EdgeDecoder(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Decodes the probability of a link by concatenating User and Movie embeddings
        self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        # Get the embeddings for the source (User) and destination (Movie)
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        z = F.relu(self.lin1(z))
        z = self.lin2(z)
        return z.view(-1)


class RecommenderGNN(nn.Module):
    def __init__(self, hidden_channels, num_users, movie_feature_dim):
        super().__init__()
        # 1. Structural Embedding for Users (They have no metadata)
        self.user_emb = nn.Embedding(num_users, hidden_channels)

        # 2. Projection Layer for Movies
        self.movie_proj = nn.Linear(movie_feature_dim, hidden_channels)

        self.encoder = HeteroGNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # Initialize node representations
        h_dict = {
            'user': self.user_emb.weight,
            'movie': self.movie_proj(x_dict['movie'])
        }

        # Pass through GraphSAGE layers
        z_dict = self.encoder(h_dict, edge_index_dict)

        # Decode the requested edges
        return self.decoder(z_dict, edge_label_index)


# In Tasks 1 and 2, you used **inductive learning**: your CNN and RNN were trained on one set of data and evaluated on completely unseen, isolated test data.
# 
# Graph machine learning often relies on **transductive learning**.
# * In a transductive setup, the model sees the *entire graph structure* (all User and Movie nodes) during the training phase.
# * Because we can't easily isolate nodes without breaking the graph, we split **edges** instead of nodes.
# * We hide a percentage of the "rates" edges to serve as our test set. The model learns by passing messages across the visible training edges, and we evaluate it by asking it to predict the existence of the edges we hid.
# 
# Regarding T.RandomLinkSplit:
# * **`num_val=0.1` and `num_test=0.1`**: We hide 10% of the rating edges for validation and 10% for final testing. The remaining 80% are our training edges.
# * **`disjoint_train_ratio=0.3`**: This is a crucial defense against data leakage. Of our 80% training edges, we separate them into two buckets. 70% are strictly used for **message passing** (building the node embeddings). The other 30% are used strictly as **supervision targets** (calculating the loss). If we let the GNN pass messages over the exact same edge it is trying to predict, it would be cheating!

# In[ ]:


import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# --- TRAINING & EVALUATION LOOP ---
def train_and_evaluate_graph(graph_data, num_epochs=40, hidden_channels=64, lr=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create Train/Val/Test Link Splits (Negative Sampling)
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        edge_types=[('user', 'rates', 'movie')],
        rev_edge_types=[('movie', 'rev_rates', 'user')]
    )

    train_data, val_data, test_data = transform(graph_data)
    train_data, val_data = train_data.to(device), val_data.to(device)

    movie_feature_dim = graph_data['movie'].x.shape[1]
    num_users = graph_data['user'].num_nodes

    # Initialize Model
    model = RecommenderGNN(hidden_channels, num_users, movie_feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    history = {'loss': [], 'val_auc': []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(train_data.x_dict,
                    train_data.edge_index_dict,
                    train_data['user', 'rates', 'movie'].edge_label_index)

        target = train_data['user', 'rates', 'movie'].edge_label.float()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(val_data.x_dict,
                            val_data.edge_index_dict,
                            val_data['user', 'rates', 'movie'].edge_label_index)

            val_pred = torch.sigmoid(val_out).cpu().numpy()
            val_target = val_data['user', 'rates', 'movie'].edge_label.cpu().numpy()

            val_auc = roc_auc_score(val_target, val_pred)
            history['val_auc'].append(val_auc)

    return history, model



# In[ ]:


num_movies = len(movie_mapping)
strategies = ['baseline', 'cv', 'nlp', 'multimodal', 'pretrained']


# In[ ]:


results = {}
for strategy in strategies:
    print(f"Training Strategy: {strategy.upper()}")

    # 1. Get features using the helper function
    features = get_movie_features(strategy, cv_features, nlp_features, num_movies)

    # 2. Build the PyG HeteroData Graph
    graph = build_graph(features)

    # 3. Train and Evaluate
    history, _ = train_and_evaluate_graph(graph, num_epochs=40)
    results[strategy] = history

# --- PLOTTING THE COMPARISON ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
styles = ['--', '-.', ':', '-', '-']
colors = ['gray', 'blue', 'green', 'purple', 'red']

for i, (name, history) in enumerate(results.items()):
    epochs = range(1, len(history['loss']) + 1)

    ax1.plot(epochs, history['loss'], linestyle=styles[i], color=colors[i], label=name.upper(), linewidth=2)
    ax2.plot(epochs, history['val_auc'], linestyle=styles[i], color=colors[i], label=name.upper(), linewidth=2)

ax1.set_title('Training Loss (BCE) over Epochs', fontsize=14)
ax1.set_xlabel('Epochs', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_title('Validation ROC-AUC over Epochs', fontsize=14)
ax2.set_xlabel('Epochs', fontsize=12)
ax2.set_ylabel('ROC-AUC Score', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()


# ---
# ### **Questions**
# 
# 1. If the movie features are random noise, how is the model still successfully predicting links? What does this tell you about the power of the Graph Structure (the edges) versus the Node Features?
# 
# 2. Is there significant differences between using one modality over the other?
# 
# 3. Did the pretrained embeddings result in better performance? Why or why not do you think that is?

# Your answers:

# ---
# ### **Qualitative analysis**
# 
# While AUC scores and loss curves are essential for measuring model performance, the true test of a recommendation system is its ability to provide tangible, human-readable results.
# 
# In this section, you will implement an **Inference Pipeline**, i.e. asking the model to predict the existence of *entirely new* edges, and printing the results.
# 
# To generate a "Top-5" list for a specific user, your code must execute the following logic:
# 
# 1.  **Extract User History**: Identify the movies User $X$ has already rated in the graph. We do this to ensure we don't recommend movies they have already seen.
# 2.  **Candidate Generation**: Identify every movie in the `movie_mapping` that does *not* appear in the user's history.
# 3.  **Link Prediction**: Construct a set of "potential edges" between User $X$ and every candidate movie. Pass these into your trained GNN to calculate a **Match Probability** (0.0 to 1.0).
# 4.  **Ranking and Mapping**: Sort the candidates by their probability scores, select the top $K$, and map the internal Graph IDs back to human-readable titles using your metadata.
# 

# In[ ]:


import torch
import pandas as pd
import numpy as np

def recommend_movies_for_user(user_id, model, graph_data, metadata_df, user_mapping, movie_mapping, top_k=5):
    """Generates human-readable movie recommendations for a specific user in the following fomat:
    --- USER X HISTORY ---
    They have already rated N movies. Here are a few:
       Movie 1
       Movie 2
       Movie 3
       Movie 4
       Movie 5


    --- TOP 5 GNN RECOMMENDATIONS ---
       Movie 1 (Probability)
       Movie 2 (Probability)
       Movie 3 (Probability)
       Movie 4 (Probability)
       Movie 5 (Probability)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    graph_data = graph_data.to(device)
    model.eval()

    # TODO: Complete the Recommendation Engine

    # 1. Reverse the `movie_mapping` dictionary so you can convert Graph IDs back to MovieLens IDs.
    #    Also, create a dictionary from `metadata_df` to look up movie titles by their MovieLens ID.

    # 2. Find the internal Graph ID for the requested `user_id` using `user_mapping`.
    #    (Optional: Add a print statement and return if the user isn't in the graph).

    # 3. Find all movies the user has ALREADY rated.
    #    Hint: Look at the full, undivided edge_index to find edges connected to this user.
    #    Print the user's history (show up to 5 movie titles they have rated).

    # 4. Find all candidate movies (Movies this user has NOT rated yet).

    # 5. Create a `pred_edge_index` tensor of shape [2, num_candidates].
    #    Row 0 should be the user's Graph ID repeating.
    #    Row 1 should be the candidate movie Graph IDs.
    #    Move this tensor to the device.

    # 6. Forward Pass: Ask the GNN to predict the probability of these candidate edges.
    #    Pass the full graph structure (edge_index_dict) to the model, so it can aggregate neighbors,
    #    but ask it to decode ONLY our new pred_edge_index.

    # 7. Sort the predictions to find the indices of the `top_k` highest probabilities.
    #    Print the final recommendations, converting the Graph IDs back to actual Movie Titles.


# In[ ]:


graph = build_graph(get_movie_features("cv", cv_features, nlp_features, num_movies))
_, model = train_and_evaluate_graph(graph, num_epochs=40)

recommend_movies_for_user(
    user_id=42,
    model=model, # Your trained GNN
    graph_data=graph, # Pass the full graph object so it has the structural context
    metadata_df=small_full, # The dataframe with movie titles
    user_mapping=user_mapping,
    movie_mapping=movie_mapping,
    top_k=5
)

