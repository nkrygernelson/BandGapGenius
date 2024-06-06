def stoichiometric_attributes(formula):
    #we calculate the the Lp norms for p = 2,3,5,7 using the stoechiometric coeffecients of the compound
    #we isolate the stoechiometric coeffecients by splitting the formula string by the numbers
    #we then convert the stoechiometric coeffecients to integers and calculate the Lp norms
    def get_coeffs(formula):
        coeffs = [] 
        tmp = ''
        for i in range(len(formula)):
            #when you get to a capital letter and the  tmp string is empty, you add 1 to the coeffs list
            #this is because the stoechiometric coefficient is 1
            #when we get a number we add it to the tmp string
            #if we are the last element in the formula string, we add the tmp string if it is non-empty to the coeffs list else we add 1
            #if we are at the first element in the formula string we do nothing

            if formula[i].isalpha():
                if formula[i].isupper():
                    if tmp == '':
                        coeffs.append(1)
                    else:
                        coeffs.append(int(tmp))
                        tmp = ''
                if i == len(formula) - 1:
                    coeffs.append(1)
            else:
                tmp += formula[i]
        if tmp != '':
            coeffs.append(int(tmp))         
        return coeffs[1:]
    
    coeffs = get_coeffs(formula)
    def Lp_norm(coeffs, p):
        return sum([abs(x/sum(coeffs))**p for x in coeffs])**(1/p)
    norms = [Lp_norm(coeffs, p) for p in [2,3,5,7]]
    return norms


    
                     


