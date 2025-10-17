import cobra
from cobra.io import load_model
from cobra.core import Metabolite, Gene, Reaction

def load_igem_model():
    model = load_model('iML1515')

    # Add metabolites to model
    model.add_metabolites([

    ])


    # Add reactions to model

    CADAtpp = Reaction(id='CADAtpp')
    CADAtex = Reaction(id='CADAtex')
    EX_15dap_e=Reaction(id='EX_15dap_e')

    model.add_reactions([CADAtpp,CADAtex,EX_15dap_e])


    # Define reaction equations
    
    CADAtpp.reaction = '1 15dap_c -> 1 15dap_p'
    CADAtex.reaction = '1 15dap_p -> 1 15dap_e'
    EX_15dap_e.reaction = '1 15dap_e ->'


    # Add genes to model
    
    

    # Define GPR rules
    

    model.id = 'F1515'
    model.name = 'F1515'

    return model
