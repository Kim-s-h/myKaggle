library(tidyverse)
library(mlr)
train <- read_csv("Titanic/train.csv")
test <- read_csv("Titanic/test.csv")

surname <- map_chr(str_split(train$Name, "\\."), 1)
salutations <- map_chr(str_split(surname, "\\, "), 2)
salutations[!(salutations %in% c("Mr", "Dr", "Master",
                                 "Miss", "Mrs", "Rev"))] <- "Other"
factorVar = c("Sex", "Pclass", "Survived", "Salutation")
titanic_train <- train %>%
  mutate(Salutation = salutations,
         FamSize = SibSp + Parch) %>%
  mutate_at(.vars = factorVar, .funs = factor) %>%
  select(Survived,Pclass, Sex, Age, FamSize, Salutation)


titanicTask <- makeClassifTask(data=titanic_train, target = "Survived")
logitWrapper <- makeImputeWrapper("classif.logreg",cols = list(Age = imputeMean()))
kFold <- makeResampleDesc(method="RepCV", folds=10, reps=50, stratify = TRUE)
logRegTitanic <- resample(logitWrapper, titanicTask, 
                          resampling = kFold, measures = list(acc, fpr, fnr))


# ---- training the data -----
imp <- impute(titanic_train, cols = list(Age = imputeMean()))
titanicTask <- makeClassifTask(data = imp$data, target = "Survived")
logReg <- makeLearner("classif.logreg", predict.type = "prob")
logRegModel <- train(logReg, titanicTask)

# ---- make predictions ----

surname <- map_chr(str_split(test$Name, "\\."), 1)
salutations <- map_chr(str_split(surname, "\\, "), 2)
salutations[!(salutations %in% c("Mr", "Dr", "Master",
                                 "Miss", "Mrs", "Rev"))] <- "Other"
factorVar = c("Sex", "Pclass", "Salutation")
titanic_test <- test %>%
  mutate(Salutation = salutations,
         FamSize = SibSp + Parch) %>%
  mutate_at(.vars = factorVar, .funs = factor) %>%
  select(Pclass, Sex, Age, FamSize, Salutation)

imp_test <- impute(titanic_test, cols = list(Age = imputeMean()))
Survived <-  predict(logRegModel, newdata = imp_test$data)

predFinal <- cbind(test,Survived) %>%
  select(PassengerId, response) %>%
  rename(Survived = response) %>%
  write_csv(path = "Titanic/submission_logit_02_20_2021.csv")
