# example 1: KNN classification with NFL field goal attempts ----
library(tidyverse)

# read in data
nfl_fg_attempts <- read_csv("https://tinyurl.com/ykr289ew") |> 
  # remove blocked field goals
  filter(field_goal_result %in% c("made", "missed"))

# separate predictors from response
fg_x <- select(nfl_fg_attempts, kick_distance, score_differential)
fg_y <- as_factor(nfl_fg_attempts$field_goal_result)

# plot data: does there appear to be an underlying trend?
#ggplot(nfl_fg_attempts) +
#  geom_point(aes(x = kick_distance, y = score_differential,
#                 color = field_goal_result), size = 0.7) + 
#  scale_color_manual(values = c("made" = "orange", "missed" = "blue")) +
#  labs(title = "Score Differential by Kick Distance",
#       x = "Kick Distance (yards)", y = "Score Differential",
#       color = "Field Goal Result") + 
#  theme_bw() + theme(plot.title = element_text(hjust = 0.5)) 

# knn time!
library(FNN)
init_knn <- knn(train = fg_x, test = fg_x, cl = fg_y,
                k = 1, algorithm = "brute")

# training and test data
set.seed(50)
train_ids <- sample(1:nrow(nfl_fg_attempts),
                    ceiling(0.75 * nrow(nfl_fg_attempts)))
train_nfl <- nfl_fg_attempts[train_ids, ] # training datasst
test_nfl <- nfl_fg_attempts[-train_ids, ] # test dataset
# separate predictors we want/care about and response
train_x <- select(train_nfl, kick_distance, score_differential)
train_y <- train_nfl$field_goal_result
test_x <- select(test_nfl, kick_distance, score_differential)
test_y <- test_nfl$field_goal_result

# assess 1NN on train data
one_nn_train_preds <- knn(train = train_x, test = train_x, 
                          cl = train_y, k = 1, algorithm = "brute")
mean(train_y == one_nn_train_preds)

# note: not 100% because there are ties!


# assess 1NN on test data
one_nn_test_preds <- knn(train = train_x, test = test_x, 
                         cl = train_y, k = 1, algorithm = "brute")
mean(test_y == one_nn_test_preds)

# performs substantially worse on test data. overfit!

# "manually" determine the value of k
errs_train <- rep(0, 12)
errs_test <- rep(0, 12)
k_vals <- 1:12

for(k in k_vals) {
  train_preds <- knn(train = train_x, test = train_x, cl = train_y, 
                     k = k, algorithm = "brute")
  errs_train[k] <- mean(!train_y == train_preds)
  
  test_preds <- knn(train = train_x, test = test_x, cl = train_y,
                    k = k, algorithm = "brute")
  errs_test[k] <- mean(!test_y == test_preds)
}
# plot train and test error
errors <- bind_cols(train_err = errs_train,
                    test_err = errs_test,
                    k = k_vals)

errors |> pivot_longer(c(train_err, test_err), 
                       names_to = "err_type") |> 
  ggplot(aes(x = k, y = value, 
             color = err_type)) +
  geom_point() + geom_line() +
  labs(y = "Misclassification Rate", 
       color = "Type of error") + theme_bw()
rm(list=ls())
gc()

# example 2: knn classification using student success data ----

# read in data. find a URL or upload the clean data (prior to scaling tho)
dat <- read_delim("raw-data/student-success.csv", 
                  delim = ";", escape_double = FALSE, trim_ws = TRUE)

# clean data
short_dat <- dat |> janitor::clean_names() |> select(13:21, 34:37)
#write_csv(short_dat, "student_success_data.csv")

#talk about why standardizing matters. don't forget to remove Y first!
clean_dat <- short_dat |> select(-target) |>  scale() |> as.data.frame()
rm(short_dat)

outcome <- dat$Target

#library(FNN)
# training and test data
set.seed(50)
train_ids <- sample(1:nrow(clean_dat), (0.8 * nrow(clean_dat)))
train_x <- clean_dat[train_ids, ]
test_x <- clean_dat[-train_ids, ]
# separate predictors and response, again
train_y <- outcome[train_ids]
test_y <- outcome[-train_ids]

# loop through to decide the value of k
errs_train <- rep(0, 12)
errs_test <- rep(0, 12)
k_vals <- 1:12

for(k in k_vals) {
  train_preds <- knn(train = train_x, test = train_x, cl = train_y, 
                     k = k, algorithm = "brute")
  errs_train[k] <- mean(!train_y == train_preds)
  
  test_preds <- knn(train = train_x, test = test_x, cl = train_y,
                    k = k, algorithm = "brute")
  errs_test[k] <- mean(!test_y == test_preds)
}
# plot train and test error
errors <- bind_cols(train_err = errs_train,
                    test_err = errs_test,
                    k = k_vals)

errors |> pivot_longer(c(train_err, test_err), 
                       names_to = "err_type") |> 
  ggplot(aes(x = k, y = value, 
             color = err_type)) +
  geom_point() + geom_line() +
  labs(y = "Misclassification Rate", 
       color = "Type of error") + theme_bw()

# fit KNN based upon 'best' value
k7nn <- knn(train = train_x, test = test_x, 
            cl = train_y, k = 7, algorithm = "brute")

#confusion matrix with more statistics
caret::confusionMatrix(k7nn, as.factor(test_y))

# extension: knn regression w/scooby doo data ----
rm(list=ls())
gc()

library(tidytuesdayR)
# read data in 
tuesdata <- tidytuesdayR::tt_load(2021, week = 29)
scoobydoo <- tuesdata$scoobydoo
#dim(scoobydoo)

# clean data
scooby_knn <- scoobydoo |> 
  filter(format == "TV Series",
         imdb != "NULL") |> 
  mutate(imdb = as.numeric(imdb)) |> 
  select(imdb, suspects_amount, monster_amount)

# knn regression time! one particular case:
k2 <- knn.reg(train = dplyr::select(scooby_knn, -imdb),
              y = scooby_knn$imdb, k = 2)

# loop through K = 1, ..., 10
knn_results <- lapply(1:10, function(k) {
  knn.reg(
    train = dplyr::select(scooby_knn, -imdb),
    y = scooby_knn$imdb,
    k = k,
    algorithm = "brute"
  )
})

# extract Rsq values
rsqvals <- rep(NA, length(knn_results))
for (i in 1:length(knn_results)){
  rsqvals[i] <- knn_results[[i]]$R2Pred
}
# turn to data frame
rsq <- data.frame(1:10, rsqvals)
colnames(rsq) <- c("k", "r_squared")
# plot Rsq values!
ggplot(rsq, aes(x = k, y = r_squared)) +
  geom_point() + geom_line() + scale_x_continuous(breaks = c(1:10)) +
  labs(
    title = "Predicted R Squared vs. Number of Neighbors K",
    x = "K", y = "Predicted R Squared"
  ) + theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5))

# verify the max
which(rsq$r_squared == max(rsq$r_squared))

# investigate the best
k5 <- knn_results[[5]]
