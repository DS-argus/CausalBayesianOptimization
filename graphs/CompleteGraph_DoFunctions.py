import sys
sys.path.append("..") 

## Import basic packages
import numpy as np



def compute_do_E(observational_samples, functions, value):
    gp_A_C_E = functions['gp_A_C_E']
    
    # observation에서 A, C가져옴 (for loop)
    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    C = np.asarray(observational_samples['C'])[:,np.newaxis]
    
    # E=e까지 붙여서 (A, C, E) 만들어줌
    intervened_inputs = np.hstack((A,C,np.repeat(value, C.shape[0])[:,np.newaxis]))

    # GP로 predict하고 평균
    mean_do = np.mean(gp_A_C_E.predict(intervened_inputs)[0])
    var_do = np.mean(gp_A_C_E.predict(intervened_inputs)[1])

    return mean_do, var_do


def compute_do_D(observational_samples, functions, value):

    gp_D_C = functions['gp_D_C']
    
    
    # observation 에서 C 가져옴
    C = np.asarray(observational_samples['C'])[:,np.newaxis]
    
    # intervention하는 D=d값과 C를 붙여서 (D, C)로 만들어줌
    intervened_inputs = np.hstack((np.repeat(value, C.shape[0])[:,np.newaxis], C))

    # GP에다가 observation 넣고 predict해서 평균
    mean_do = np.mean(gp_D_C.predict(intervened_inputs)[0])
    var_do = np.mean(gp_D_C.predict(intervened_inputs)[1])

    return mean_do, var_do
   


def compute_do_B(observational_samples, functions, value):
    
    gp_C = functions['gp_C']
    gp_B_C = functions['gp_B_C']
    
    # b'에 대해서 for loop 도니 B 가져옴
    B = np.asarray(observational_samples['B'])[:,np.newaxis]
    
    # B=b를 input으로해서 c predict
    intervened_inputs = np.repeat(value, B.shape[0])[:,np.newaxis]
    new_c = np.mean(gp_C.predict(intervened_inputs)[0])
    
    # 위에서 만든 C를 붙여서 (B, C)로 만들어줌
    intervened_inputs2 = np.hstack((B, np.repeat(new_c, B.shape[0])[:,np.newaxis]))

    # GP로 predict
    mean_do = np.mean(gp_B_C.predict(intervened_inputs2)[0])
    var_do = np.mean(gp_B_C.predict(intervened_inputs2)[1])

    return mean_do, var_do


def compute_do_DE(observational_samples, functions, value):

    gp_D_E_C_A = functions['gp_D_E_C_A']

    # for loop 도는 A, C가져옴
    C = np.asarray(observational_samples['C'])[:,np.newaxis]
    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    
    # (D=d, E=e, C, A) 만들어줌
    intervened_inputs = np.hstack((np.repeat(value[0], C.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], C.shape[0])[:,np.newaxis], 
                                   C, A))

    # GP로 predict
    mean_do = np.mean(gp_D_E_C_A.predict(intervened_inputs)[0])
    var_do = np.mean(gp_D_E_C_A.predict(intervened_inputs)[1])

    return mean_do, var_do

   
def compute_do_BE(observational_samples, functions, value):

    gp_C = functions['gp_C']
    gp_B_E_C_A = functions['gp_B_E_C_A']
    
    # B', A 가져옴
    B = np.asarray(observational_samples['B'])[:,np.newaxis]
    A = np.asarray(observational_samples['A'])[:,np.newaxis]

    # B=b 로 C predict
    intervened_inputs = np.repeat(value[0], B.shape[0])[:,np.newaxis]
    new_c = np.mean(gp_C.predict(intervened_inputs)[0])
    
    # (B', E, C, A) 만들어줌
    intervened_inputs2 = np.hstack((B, np.repeat(value[1], B.shape[0])[:,np.newaxis],
                                   np.repeat(new_c, B.shape[0])[:,np.newaxis], A))
    
    # predict
    mean_do = np.mean(gp_B_E_C_A.predict(intervened_inputs2)[0])
    var_do = np.mean(gp_B_E_C_A.predict(intervened_inputs2)[1])

    return mean_do, var_do


def compute_do_BD(observational_samples, functions, value): 

    gp_C = functions['gp_C']
    gp_B_C_D = functions['gp_B_C_D']
    
    # B' 가져옴
    B = np.asarray(observational_samples['B'])[:,np.newaxis]

    # B=b 로 C predict
    intervened_inputs = np.repeat(value[0], B.shape[0])[:,np.newaxis]
    new_c = np.mean(gp_C.predict(intervened_inputs)[0])
    
    # (B', C, D) 만들어줌
    intervened_inputs2 = np.hstack((B, np.repeat(new_c, B.shape[0])[:,np.newaxis],
                                    np.repeat(value[1], B.shape[0])[:,np.newaxis]))
    
    # GP로 predict
    mean_do = np.mean(gp_B_C_D.predict(intervened_inputs2)[0])
    var_do = np.mean(gp_B_C_D.predict(intervened_inputs2)[1])

    return mean_do, var_do
   
def compute_do_BDE(observational_samples, functions, value):   
    
    gp_D_E_C_A = functions['gp_D_E_C_A']

    C = np.asarray(observational_samples['C'])[:,np.newaxis]
    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    
    intervened_inputs = np.hstack((np.repeat(value[0], C.shape[0])[:,np.newaxis], 
                                   np.repeat(value[1], C.shape[0])[:,np.newaxis], 
                                   C, A))
    mean_do = np.mean(gp_D_E_C_A.predict(intervened_inputs)[0])
    
    var_do = np.mean(gp_D_E_C_A.predict(intervened_inputs)[1])

    return mean_do, var_do
    

def compute_do_BDEF(observational_samples, functions, value):   
    
    gp_C = functions['gp_C']
    gp_A_B_C_D_E_F = functions['gp_A_B_C_D_E_F']
    gp_A = functions['gp_A']

    
    A = np.asarray(observational_samples['A'])[:,np.newaxis]
    B = np.asarray(observational_samples['B'])[:,np.newaxis]
    F = np.asarray(observational_samples['F'])[:,np.newaxis]
    
    intervened_inputs = np.repeat(value[0], B.shape[0])[:,np.newaxis]
    new_c = np.mean(gp_C.predict(intervened_inputs)[0])
    
 
    intervened_inputs = np.repeat(value[3], B.shape[0])[:,np.newaxis]
    new_a = np.mean(gp_A.predict(intervened_inputs)[0])
    
    
    intervened_inputs2 = np.hstack((np.repeat(new_a, B.shape[0])[:,np.newaxis], 
                                    B, 
                                    np.repeat(new_c, B.shape[0])[:,np.newaxis],
                                    np.repeat(value[2], B.shape[0])[:,np.newaxis],
                                    np.repeat(value[1], B.shape[0])[:,np.newaxis],
                                    F))
    
    
    mean_do = np.mean(gp_A_B_C_D_E_F.predict(intervened_inputs2)[0])
    
    var_do = np.mean(gp_A_B_C_D_E_F.predict(intervened_inputs2)[1])

    return mean_do, var_do

