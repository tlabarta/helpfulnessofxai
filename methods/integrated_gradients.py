from locale import format_string
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
import torch
from copy import deepcopy


class IntegratedGradientsExplainer():


    def __init__(self, model):
        self.model = model


    def explain(self, input_tensor):
        input_tensor = deepcopy(input_tensor)
        output = self.model.predict(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        self.out_idx = int(probabilities.detach().numpy().argmax())
        input_tensor.requires_grad = True # inplace? create copy of tensor before setting requires_grad?
        
        ig = IntegratedGradients(self.model)
        attr_ig, delta = self.__attribute_image_features(ig, input_tensor, baselines=input_tensor * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print('Approximation delta: ', abs(delta))


        # Plot Overlayed Integrated Gradients
        original_image = np.transpose(input_tensor.cpu().detach().numpy()[0] , (1, 2, 0))# / 2) + 0.5
        fig, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                                show_colorbar=True, cmap="bwr", title=None, use_pyplot=False)
        
        return fig
        

    def __attribute_image_features(self, algorithm, input, **kwargs):
        self.model.model.zero_grad() # inplace? create copy of model before setting zero_grad()?
        tensor_attributions = algorithm.attribute(input,
                                                target=int(self.out_idx),
                                                **kwargs
                                                )
        
        return tensor_attributions



