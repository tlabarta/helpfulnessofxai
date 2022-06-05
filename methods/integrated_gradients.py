from locale import format_string
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import numpy as np
import torch


class IntegratedGradientsExplainer():


    def __init__(self, model):
        self.model = model


    def explain(self, input_tensor, file_name):
        output = self.model.predict(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        self.out_idx = int(probabilities.detach().numpy().argmax())
        input_tensor.requires_grad = True # inplace? create copy of tensor before setting requires_grad?
        
        ## Integrated Gradients
        ig = IntegratedGradients(self.model)
        attr_ig, delta = self.__attribute_image_features(ig, input_tensor, baselines=input_tensor * 0, return_convergence_delta=True)
        attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
        print('Approximation delta: ', abs(delta))

        ## Integrated Gradients with SmoothGrad Squared
        # ig = IntegratedGradients(self.model)
        # nt = NoiseTunnel(ig)
        # attr_ig_nt = self.__attribute_image_features(nt, input_tensor, baselines=input_tensor * 0, nt_type='smoothgrad_sq',
        #                                     nt_samples=100, stdevs=0.2)
        # attr_ig_nt = np.transpose(attr_ig_nt.squeeze(0).cpu().detach().numpy(), (1, 2, 0))

        # Plot Overlayed Integrated Gradients
        original_image = np.transpose(input_tensor.cpu().detach().numpy()[0] , (1, 2, 0))# / 2) + 0.5
        fig, _ = viz.visualize_image_attr(attr_ig, original_image, method="blended_heat_map",sign="all",
                                show_colorbar=True, cmap="Purples", title="Overlayed Integrated Gradients", use_pyplot=False)
        
        # save figure
        # image_name = file_name.split(".")[0]
        # format_str = file_name.split(".")[1]
        # output_path = f"results/IntegratedGradients/{image_name}_{self.model.name}.{format_str}"
        # fig.savefig(output_path)
        return fig
        

    def __attribute_image_features(self, algorithm, input, **kwargs):
        self.model.model.zero_grad() # inplace? create copy of model before setting zero_grad()?
        tensor_attributions = algorithm.attribute(input,
                                                target=int(self.out_idx),
                                                **kwargs
                                                )
        
        return tensor_attributions



