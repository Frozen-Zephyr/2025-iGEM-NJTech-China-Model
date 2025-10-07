import cobra
from cobra.io import load_model
from cobra.core import Metabolite, Gene, Reaction

def load_igem_model():
    model = load_model('iML15151')

    # Add metabolites to model
    model.add_metabolites([

    ])


    # Add reactions to model

    CADAtpp1 = Reaction(id='CADAtpp1')
    CADAtex1 = Reaction(id='CADAtex1')
    EX_succ_e=Reaction(id='EX_succ_e')

    model.add_reactions([CADAtpp1,CADAtex1,EX_succ_e])


    # Define reaction equations
    
    CADAtpp1.reaction = '1 succ_c -> 1 succ_p'
    CADAtex1.reaction = '1 succ_p -> 1 succ_e'
    EX_succ_e.reaction = '1 succ_e ->'


    # Add genes to model
    
    

    # Define GPR rules
    

    model.id = 'F15151'
    model.name = 'F15151'

    return model
