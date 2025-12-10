library(tidyverse)
library(tidymodels)
library(embed)
library(dplyr)
library(vroom)
library(DataExplorer)
library(GGally)
library(patchwork)
library(glmnet)
library(discrim)
library(kernlab)
library(themis)
library(prophet)

############################################################################

# Reading in the Data
testData <- vroom("test.csv")
trainData <- vroom("train.csv")
features <- vroom("features.csv")
stores <- vroom("stores.csv")

# Updating Features
features <- features %>%
  mutate(across(starts_with("Markdown"), ~ replace_na(., 0))) %>%
  mutate(TotalMarkdown = rowSums(across(starts_with("Markdown")))) %>%
  mutate(MarkdownFlag = as.numeric(TotalMarkdown > 0)) %>%
  select(- c(MarkDown1, MarkDown2, MarkDown3, MarkDown4, MarkDown5))

feature_recipe <- recipe(~., data=features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))
imputed_features <- juice(prep(feature_recipe))

# Joining Data with Features
new_trainData <- left_join(trainData, imputed_features, by = c("Store", "Date"))
new_testData <- left_join(testData, imputed_features, by = c("Store", "Date"))

filtered_data <- new_trainData %>%
  group_by(Store, Dept) %>%
  filter(n() > 10) %>%
  ungroup()

############################################################################

# Recipe
filtered_data <- filtered_data |>
  mutate(IsHoliday = IsHoliday.x) |>
  select(-IsHoliday.y, - IsHoliday.x)

new_testData <- new_testData |>
  mutate(IsHoliday = IsHoliday.x) |>
  select(-IsHoliday.y, - IsHoliday.x)
  
my_recipe <- recipe(Weekly_Sales ~ ., data = filtered_data) %>%
  step_mutate(IsHoliday = as.factor(IsHoliday)) %>%
  step_date(Date, features =c("year", "week", "dow")) %>%
  step_rm(Date) %>%
  step_dummy(all_nominal_predictors())

  
prepped_recipe <- prep(my_recipe)
baked_data <- bake(prepped_recipe, new_data=filtered_data)

############################################################################

# Setting Up Some Models
# Penalized Linear Regression
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% 
                         set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5) 

folds <- vfold_cv(filtered_data, v = 5, repeats=1)

CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

collect_metrics(CV_results) %>% 
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

bestTune <- CV_results %>%
  select_best(metric="rmse")

best_rmse <- show_best(CV_results, metric = "rmse", n = 1) %>%
  pull(mean)

final_wf <-preg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=filtered_data)

final_wf %>%
  predict(new_data = new_testData)

walmart_predictions <- predict(final_wf, new_data=new_testData)

############################################################################

small_data <- filtered_data |>
  filter(Store == 1, Dept == 2)
  
  
small_testData <- new_testData |>
  filter(Store == 1, Dept == 2)
  
  
forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=100) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")


# Create a workflow with model & recipe
forest_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(forest_mod)

grid_of_tuning_params <- grid_regular(mtry(range = c(1, 11)),
                                      min_n(range = c(2,20)),
                                      levels = 5)

folds <- vfold_cv(small_data, v = 5, repeats=1)

# Set up grid of tuning values
CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=grid_of_tuning_params,
            metrics=metric_set(rmse, mae))

# Set up K-fold CV
bestTune <- CV_results %>%
  select_best(metric="rmse")

best_rmse <- show_best(CV_results, metric = "rmse", n = 1) %>%
  pull(mean)

# Finalize workflow and predict
final_wf <-forest_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=small_data)

walmart_predictions <- final_wf %>%
  predict(new_data = small_testData)

############################################################################

# Facebook Prophet Model

store <- 1 # I did 17
dept <- 1 # I used 17

sd_train_1 <- filtered_data %>%
  filter(Store==store, Dept==dept) %>%
  rename(y=Weekly_Sales, ds=Date)

sd_test_1 <- new_testData %>%
  filter(Store==store, Dept==dept) %>%
  rename(ds=Date)

prophet_model_1 <- prophet() %>%
  add_regressor("CPI") %>%
  add_regressor("Unemployment") %>%
  add_regressor("TotalMarkdown") %>%
  fit.prophet(df=sd_train_1)

fitted_vals_1 <- predict(prophet_model_1, df=sd_train_1)
test_preds_1 <- predict(prophet_model_1, df=sd_test_1)

store <- 17 # I did 17
dept <- 17 # I used 17

sd_train_17 <- filtered_data %>%
  filter(Store==store, Dept==dept) %>%
  rename(y=Weekly_Sales, ds=Date)

sd_test_17 <- new_testData %>%
  filter(Store==store, Dept==dept) %>%
  rename(ds=Date)

prophet_model_17 <- prophet() %>%
  add_regressor("CPI") %>%
  add_regressor("Unemployment") %>%
  add_regressor("TotalMarkdown") %>%
  fit.prophet(df=sd_train_17)

fitted_vals_17 <- predict(prophet_model, df=sd_train_17)
test_preds_17 <- predict(prophet_model, df=sd_test_17)

p1 <- ggplot() +
  geom_line(data = sd_train_1, mapping = aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals_1, mapping = aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds_1, mapping = aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c("Data" = "black", "Fitted" = "blue", "Forecast" = "red")) +
  labs(color="")

p2 <- ggplot() +
  geom_line(data = sd_train_17, mapping = aes(x = ds, y = y, color = "Data")) +
  geom_line(data = fitted_vals_17, mapping = aes(x = as.Date(ds), y = yhat, color = "Fitted")) +
  geom_line(data = test_preds_17, mapping = aes(x = as.Date(ds), y = yhat, color = "Forecast")) +
  scale_color_manual(values = c("Data" = "black", "Fitted" = "blue", "Forecast" = "red")) +
  labs(color="")

p1 + p2 + plot_layout(ncol = 2)


# kaggle_submission <- test_preds %>%
#   bind_cols(., sd_test) %>%
#   mutate(Weekly_Sales = yhat) %>%
#   mutate(ID = paste(1, 1, ds...31, sep = "_")) %>%
#   select(ID, Weekly_Sales)
# vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")


############################################################################


# Kaggle Submission

kaggle_submission <- walmart_predictions %>%
  bind_cols(., new_testData) %>%
  mutate(Weekly_Sales = .pred) %>%
  mutate(ID = paste(Store, Dept, Date, sep = "_")) %>%
  select(ID, Weekly_Sales)
vroom_write(kaggle_submission, file = "./LinearPreds.csv", delim = ",")

############################################################################

################
# Dr. Heaton's #
#     Code     #
################

## Libraries I need
library(tidyverse)
library(vroom)
library(tidymodels)
library(DataExplorer)

## Read in the Data
train <- vroom("./train.csv")
test <- vroom("./test.csv")
features <- vroom("./features.csv")

#########
## EDA ##
#########
plot_missing(features)
plot_missing(test)

### Impute Missing Markdowns
features <- features %>%
  mutate(across(starts_with("MarkDown"), ~ replace_na(., 0))) %>%
  mutate(across(starts_with("MarkDown"), ~ pmax(., 0))) %>%
  mutate(
    MarkDown_Total = rowSums(across(starts_with("MarkDown")), na.rm = TRUE),
    MarkDown_Flag = if_else(MarkDown_Total > 0, 1, 0),
    MarkDown_Log   = log1p(MarkDown_Total)
  ) %>%
  select(-MarkDown1, -MarkDown2, -MarkDown3, -MarkDown4, -MarkDown5)

## Impute Missing CPI and Unemployment
feature_recipe <- recipe(~., data=features) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  step_impute_bag(CPI, Unemployment,
                  impute_with = imp_vars(DecDate, Store))
imputed_features <- juice(prep(feature_recipe))

########################
## Merge the Datasets ##
########################

fullTrain <- left_join(train, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday=IsHoliday.x) %>%
  select(-MarkDown_Total)
fullTest <- left_join(test, imputed_features, by=c("Store", "Date")) %>%
  select(-IsHoliday.y) %>%
  rename(IsHoliday=IsHoliday.x) %>%
  select(-MarkDown_Total)
plot_missing(fullTrain)
plot_missing(fullTest)

##################################
## Loop Through the Store-Depts ## 
## and generate predictions.    ##
##################################
all_preds <- tibble(Id = character(), Weekly_Sales = numeric())
n_storeDepts <- fullTest %>% distinct(Store, Dept) %>% nrow()
cntr <- 0
for(store in unique(fullTest$Store)){
  
  store_train <- fullTrain %>%
    filter(Store==store)
  store_test <- fullTest %>%
    filter(Store==store)
  
  for(dept in unique(store_test$Dept)){
    
    ## Filter Test and Training Data
    dept_train <- store_train %>%
      filter(Dept==dept)
    dept_test <- store_test %>%
      filter(Dept==dept)
    
    ## If Statements for data scenarios
    if(nrow(dept_train)==0){
      
      ## Predict 0
      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
                  Weekly_Sales=0)
      
    } else if(nrow(dept_train) < 10 && nrow(dept_train) > 0){
      
      ## Predict the mean
      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
                  Weekly_Sales=mean(dept_train$Weekly_Sales))
      
    } else {
      
      ## Fit a penalized regression model
      my_recipe <- recipe(Weekly_Sales ~ ., data = dept_train) %>%
        step_mutate(Holiday = as.integer(IsHoliday)) %>%
        step_date(Date, features=c("month","year")) %>%
        step_rm(Date, Store, Dept, IsHoliday)
      prepped_recipe <- prep(my_recipe)
      tst <- bake(prepped_recipe, new_data=dept_test)
      
      my_model <- rand_forest(mtry=9,
                              trees=500,
                              min_n=5) %>%
        set_engine("ranger") %>%
        set_mode("regression")
      
      my_wf <- workflow() %>%
        add_recipe(my_recipe) %>%
        add_model(my_model) %>%
        fit(dept_train)
      
      preds <- dept_test %>%
        transmute(Id=paste(Store, Dept, Date, sep="_"),
                  Weekly_Sales=predict(my_wf, new_data = .) %>%
                    pull(.pred))
      
    }
    
    ## Bind predictions together
    all_preds <- bind_rows(all_preds,
                           preds)
    
    ## Print out Progress
    cntr <- cntr+1
    cat("Store", store, "Department", dept, "Completed.",
        round(100 * cntr / n_storeDepts, 1), "% overall complete.\n")
    
  } ## End Dept Loop
  
} ## End Store Loop

## Write out after each store so I don't have to start over
vroom_write(x=all_preds, 
            file=paste0("./Predictions.csv"), delim=",")

