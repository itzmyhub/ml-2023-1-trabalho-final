from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from codify_and_representing_data import codify_and_representing_data


X_train, X_test, y_train, y_test = codify_and_representing_data()

classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Fazer previsões nos dados de teste
predictions = classifier.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, predictions)
print(f"Precisão: {accuracy}")
