import torch

class LoroLinear(torch.nn.Module):
    def __init__(self, auto_mode=True, input_feature_dim=None, output_feature_dim=None, original_linear=None, bias=True, r=8, device='cuda', noise_mag=1e-1):
        super().__init__()
        
        if auto_mode: 
            self.input_feature_dim = original_linear.in_features
            self.output_feature_dim = original_linear.out_features
            self.bias = original_linear.bias is not None
        else:
            self.input_feature_dim = input_feature_dim
            self.output_feature_dim = output_feature_dim
            self.bias = bias
        
        self.deobfus_inference = True 
        self.r = 8
        
        self.otp = torch.rand(self.input_feature_dim) * noise_mag
        self.otp = self.otp.to(device)

        self.B = torch.rand(self.input_feature_dim, self.r) * noise_mag
        self.B = self.B.to(device)
        self.A = torch.rand(self.r, self.output_feature_dim) * noise_mag
        self.A = self.A.to(device)
        
        original_weight = original_linear.weight # [output_dim, input_dim]才是线性层的参数shape
        self.obfus_linear = torch.nn.Linear(self.input_feature_dim, self.output_feature_dim, bias=bias).to(device)
        self.obfus_linear.weight = torch.nn.Parameter((self.B@self.A).T + original_weight.to(device))
        if self.bias == True:
            self.obfus_linear.bias = torch.nn.Parameter(original_linear.bias.to(device))
        
        self.pre_compute = self.otp @ self.obfus_linear.weight.T
         
    def forward(self, x):
        
        if self.deobfus_inference: # 如果要看解混淆的准确率
            real_time_compute = x @ self.B @ self.A
            x = x + self.otp # 加掩码
            x = self.obfus_linear(x) # 混淆参数推理
            x = x - real_time_compute - self.pre_compute # 解混淆
        else: # 如果要看混淆模型准确率下降到啥样
            x = self.obfus_linear(x)
        return x