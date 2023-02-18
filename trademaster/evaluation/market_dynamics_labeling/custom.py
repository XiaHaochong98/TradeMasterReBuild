from .builder import Market_Dynamics_Model


@Market_Dynamics_Model.register_module()
class Market_dynamics_model():
    def __init__(self, **kwargs):
        super(Market_Dynamics_Model, self).__init__()