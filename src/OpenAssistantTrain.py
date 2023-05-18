import RetrainNetwork
import torch
from OpenAssistantDataset import OpenAssistantDataset
def main():
    """
    This function fine tunes the model on Open Assistant data
    :return: Nothing

    Saves the state dict of the model
    """
    DATA_ROOT = "../data/OpenAssistantConversations"
    OAtrain = OpenAssistantDataset('train', DATA_ROOT + 'train.npy')

    OAval = OpenAssistantDataset('val', DATA_ROOT + 'validation.npy')

    MODEL_PATH = '../models'
    model = torch.load(MODEL_PATH + "/BaseHopfieldNetwork.pth")

    RetrainNetwork.main([OAtrain, OAval], model)
    torch.save(model.state_dict(), MODEL_PATH + '/OAHopfieldNetwork.pth')

if __name__ == '__main__':
    main()

