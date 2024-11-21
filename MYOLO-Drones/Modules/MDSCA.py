class MDSCA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(MDSCA, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.sg_attention_branch = SGAttention(dim=in_planes // 2)

        self.channel_attention = CGA(in_planes // 2)

        self.ConvLinear = BasicConv(in_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        half_channel = x.shape[1] // 2
        x1 = x[:, :half_channel, :, :]
        x2 = x[:, half_channel:, :, :]

        x1 = self.sg_attention_branch(x1)

        x2 = self.channel_attention(x2)

        combined = torch.cat((x1, x2), 1)

        out = self.ConvLinear(combined)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SGAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 group_kernel_sizes: t.List[int] = [3, 5, 3, 5],
                 gate_layer: str = 'sigmoid'):

        super(SGAttention, self).__init__()
        self.dim = dim
        self.group_chans = dim // 4


        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=self.group_chans)

        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=3,
                                      padding=group_kernel_sizes[2] // 2, groups=self.group_chans, dilation=1)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, kernel_size=5,
                                      padding=group_kernel_sizes[3] // 2, groups=self.group_chans, dilation=1)

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv1x1 = nn.Conv1d(dim, dim, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        b, c, h_, w_ = x.size()

        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))

        x_h_attn = self.conv1x1(x_h_attn)
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = self.conv1x1(x_w_attn)
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        return x * x_h_attn * x_w_attn


class CGA(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., ratio=16):
        super(CGA, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.ca_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_max_pool = nn.AdaptiveMaxPool2d(1)
        self.ca_fc1 = nn.Conv2d(in_features, in_features // ratio, 1, bias=False)
        self.ca_relu1 = nn.ReLU()
        self.ca_fc2 = nn.Conv2d(in_features // ratio, in_features, 1, bias=False)
        self.ca_sigmoid = nn.Sigmoid()

        self.h_conv = nn.Conv2d(in_features, in_features, (1, 3), padding=(0, 1), groups=in_features)
        self.v_conv = nn.Conv2d(in_features, in_features, (3, 1), padding=(1, 0), groups=in_features)

        self.g_fn = GFFN(in_features, hidden_features, out_features, act_layer, drop)

    def forward(self, x):
        B, C, H, W = x.shape
        avg_x = self.ca_avg_pool(x)
        avg_x_h = self.h_conv(avg_x)
        avg_x_v = self.v_conv(avg_x_h)
        avg_x = avg_x_v
        avg_x = self.ca_relu1(avg_x)

        # Max Pooling branch
        max_x = self.ca_max_pool(x)
        # max_x = self.ca_fc1(max_x)

        # Apply horizontal convolution first, then vertical convolution
        max_x_h = self.h_conv(max_x)
        max_x_v = self.v_conv(max_x_h)  # Vertical convolution after horizontal convolution
        max_x = max_x_v
        max_x = self.ca_relu1(max_x)
        # max_x = self.ca_fc2(max_x)

        # Combine avg and max outputs before applying sigmoid
        ca_out = avg_x + max_x
        ca_out = self.ca_sigmoid(ca_out)  # Apply sigmoid after combining

        x = ca_out * x

        # Reshape for GFFN
        x = x.view(B, C, H * W).transpose(1, 2)  # Change shape to (B, H*W, C)

        # GFFN
        x = self.g_fn(x, H, W)

        # Reshape back to original shape if needed
        x = x.transpose(1, 2).view(B, C, H, W)
        return x


class GFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(GFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.norm = nn.LayerNorm(hidden_features // 2)
        self.conv = nn.Conv2d(hidden_features // 2, hidden_features // 2, kernel_size=3, stride=1, padding=1,
                              groups=hidden_features // 2)  # DW Conv
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # Split
        x1, x2 = x.chunk(2, dim=-1)
        # H = int(N ** 0.5)  # Assuming N is a perfect square
        # W = N // H
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()
        x = x1 * x2
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x