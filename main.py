from utils.logistic import LogisticReg

[X, y] = LogisticReg.loadData('./data/data-logistic.csv')
clf = LogisticReg(f_number=2, c=10, k=0.1)
clf.fit(X, y)
mark_without_reg = clf.getassesment(X,y)
clf.fit(X, y, regulariz=True)
mark_with_reg = clf.getassesment(X, y)

'''print(mark_without_reg, mark_with_reg)
with open('marks.txt', 'w') as f:
    f.write(f'{mark_without_reg:2.3f} {mark_with_reg:2.3f}')'''
