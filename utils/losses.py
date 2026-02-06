import torch.nn.functional as F

def distillation_loss(new_logits, old_logits, T=2.0):
    """
    T (Temperature) softens the probabilities to transfer 'dark knowledge' 
    from the old model to the new one.
    """
    outputs = F.log_softmax(new_logits / T, dim=1)
    labels = F.softmax(old_logits / T, dim=1)
    return F.kl_div(outputs, labels, reduction='batchmean') * (T**2)