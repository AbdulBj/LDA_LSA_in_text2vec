library(stringr)
library(text2vec)

#load data
data("movie_review")
# select 1000 rows for faster running times
movie_review_train = movie_review[1:700, ]
movie_review_test = movie_review[701:1000, ]
prep_fun = function(x) {
  # make text lower case
  x = str_to_lower(x)
  # remove non-alphanumeric symbols
  x = str_replace_all(x, "[^[:alpha:]]", " ")
  # collapse multiple spaces
  x = str_replace_all(x, "\\s+", " ")
}
movie_review_train$review = prep_fun(movie_review_train$review)
it = itoken(movie_review_train$review, progressbar = FALSE)
v = create_vocabulary(it)
v = prune_vocabulary(v, doc_proportion_max = 0.1, term_count_min = 5)
vectorizer = vocab_vectorizer(v)
dtm = create_dtm(it, vectorizer)

# tf-idf scaling and model fitting
tfidf = TfIdf$new()
lsa = LSA$new(n_topics = 10)

# pipe friendly transformation
doc_embeddings =  fit_transform(dtm, tfidf)
doc_embeddings =  fit_transform(doc_embeddings, lsa)

dim(doc_embeddings)

# lets test the model
it = itoken(movie_review_test$review, preprocessor = prep_fun, progressbar = FALSE)
new_doc_embeddings = create_dtm(it, vectorizer)
# apply exaxtly same scaling wcich was used in train data
new_doc_embeddings = transform(new_doc_embeddings, tfidf)
# embed into same space as was in train data
new_doc_embeddings = transform(new_doc_embeddings, lsa)

dim(new_doc_embeddings)
