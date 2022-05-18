def new_cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.loss_record = None
    cls.init_weights()