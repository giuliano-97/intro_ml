# Task 1b
Branch **junota** contains Jurij's implementation of task 1b. The following improvements should be made:
* [x]  The code should print to an output file (the output part is still not implemented yet).
* [x]  Try with standardization of X,y and then destandardization.
* [ ]  Try with RidgeCV and LassoCV with more intemediate lambdas.
* [x]  Try without fit intercept and the 21st constant feature (one should reconstruct the value of the 21st coefficient from the destandardization process).

Please update this list with all the relevant informations, intuitions et similia.

`#Retrieve parameter`\
`print(reg.coef_)`\
`print(reg.intercept_)`\
`print(reg.alpha_)`\
`print("\n")`\
`i = 0`\
`for row in reg.mse_path_:`\
`   print("{} {}".format(reg.alphas_[i],np.mean(row)))`\
`   i+=1`