

def dice_loss(prediction,ground_truth):
            
    smooth = 1.
    iflat = prediction.view(-1)
    tflat = ground_truth.view(-1)
    intersection = (iflat * tflat).sum()
  
    loss = 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
    return loss

def dice_score(prediction,ground_truth,):
    
    smooth = 1.
    iflat = prediction.view(-1)
    tflat = ground_truth.view(-1)
    intersection = (iflat * tflat).sum()  
    score =  ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))
    
    return score