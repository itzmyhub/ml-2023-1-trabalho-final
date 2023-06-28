import matplotlib.pyplot as plt

algorithms = ['Decision Tree', 'SVM', 'Naive Bayes', 'Logistic Regression']
accuracy_scores = [0.644973968762515, 0.7615138165798959, 0.6774128954745695, 0.76431718061674]

plt.bar(algorithms, accuracy_scores)
plt.xlabel('Algoritmos')
plt.ylabel('Precisão')
plt.title('Comparação dos Algoritmos de Classificação')
plt.ylim(0, 1)
plt.show()