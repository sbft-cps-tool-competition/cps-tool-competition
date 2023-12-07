
import torch
import torch.utils.data

from torchvision import transforms

import numpy as np
from ambiegenvae.generators.kappa_generator import KappaRoadGenerator
from ambiegenvae.generators.abstract_generator import AbstractGenerator
from ambiegenvae.common.train_vae import VecVAE, Denormalize1D, Normalize1D, ToTensor1D


def load_model(epoch, model, path='.//'):

    # creating the file name indexed by the epoch value
    filename = path + '/neural_network_{}.pt'.format(epoch)

    # loading the parameters of the saved model
    model.load_state_dict(torch.load(filename))

    return model

def sample_model(num_samples, model, device, mean, std):
    num_samples = num_samples  # You can change the number of samples as needed
    latent_space_size = model.latent_space

    z_samples = torch.randn(num_samples, latent_space_size).to(device)

    transform = Denormalize1D(mean, std)
    model.eval()
    with torch.no_grad():
        generated_data = model.decode(z_samples)

    scenarios = []
    for t in generated_data:
        t = transform(t.cpu())
        t = t.cpu().numpy()
        scenarios.append(t)

    return scenarios


def sample_dataset(num_samples, model, device, dataset:np.ndarray, gen:AbstractGenerator, save_path:str):
    num_samples = num_samples  # You can change the number of samples as needed

    model.eval()
    mean = np.mean(dataset, axis=0)  # mean for each feature
    std = np.std(dataset, axis=0)

    device = device

    transformations = transforms.Compose([ToTensor1D(), Normalize1D(mean, std)])
    de_transform = Denormalize1D(mean, std)

    for i in range(num_samples):
        kappas = dataset[i]
        road_points = gen.kappas_to_road_points(kappas)
        gen.visualize_test(road_points, save_path=save_path + "_original_images", num=i)
        input_data = transformations(kappas)
        input_data = input_data.unsqueeze(0)
        input_data = input_data.to(device)
        with torch.no_grad():
            result, mu, logvar = model(input_data)
        result = de_transform(result.cpu())
        result = result.squeeze(0).cpu().numpy()
        
        road_points2 = gen.kappas_to_road_points(result)
        gen.visualize_test(road_points2, save_path=save_path + "_vae_images", num=i, title=f"mu {mu.cpu().numpy()} logvar {logvar.cpu().numpy()}")


if __name__ == "__main__":
    nDim = 17 # Dimension of the input vector for VAE
    nLat = 17 # Dimension of the latent space for VAE

    archive = np.load("vae_dataset.npy")

    mean = np.mean(archive, axis=0)  # mean for each feature
    std = np.std(archive, axis=0)
    model = VecVAE(nDim, nLat)

    mod_name = 'VAE_moodels'
    model = load_model(1500, model, path=mod_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen =  KappaRoadGenerator(200, solution_size=20)


    sample_dataset(100, model, device, archive, gen, mod_name) # for visualizing the collected dataset (original and decoded results)

    ''' 
    # for visualizing the generated scenarios
    scenarios = sample_model(200, model, device, mean, std)
    gen =  KappaRoadGenerator(200)
    i = 0
    for scenario in scenarios:
        road_points = gen.kappas_to_road_points(scenario)
        gen.visualize_test(road_points, save_path=mod_name + "images\\", num=i)
        i += 1
        #time.sleep(0.5)
        print(scenario)
    '''
