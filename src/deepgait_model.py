import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalConv2d(nn.Module):
    """GaitPart-style focal convolution: split feature map horizontally, convolve each stripe, then concatenate."""
    def __init__(self, in_channels, out_channels, kernel_size=3, parts=4, padding=1):
        super().__init__()
        if parts not in (1, 2, 4, 8, 16):
            raise ValueError("parts must be a power-of-two style horizontal split")
        self.parts = parts
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        if self.parts == 1:
            return self.conv(x)
        h = x.shape[-2]
        if h % self.parts != 0:
            raise ValueError(f"height {h} is not divisible by parts {self.parts}")
        stripes = torch.chunk(x, self.parts, dim=-2)
        return torch.cat([self.conv(s) for s in stripes], dim=-2)


class GenerateNetwork(nn.Module):
    """GGCN low-level supervised mapping stage. NM/BG/CL use separate 5x5 conv branches."""
    def __init__(self):
        super().__init__()
        def branch():
            return nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        self.nm = branch()
        self.bg = branch()
        self.cl = branch()

    def _run_branch(self, x, net):
        # x: [B,T,1,H,W]
        b,t,c,h,w=x.shape
        y=net(x.reshape(b*t,c,h,w))
        return y.reshape(b,t,*y.shape[1:])

    def forward(self, nm=None, bg=None, cl=None):
        out={}
        if nm is not None: out["nm"]=self._run_branch(nm,self.nm)
        if bg is not None: out["bg"]=self._run_branch(bg,self.bg)
        if cl is not None: out["cl"]=self._run_branch(cl,self.cl)
        return out


class EncoderNetwork(nn.Module):
    """Table-2 encoder from GGCN: FConv(4,4) -> pool -> FConv(8,8) -> pool."""
    def __init__(self):
        super().__init__()
        self.f1=FocalConv2d(32,64,3,parts=4,padding=1)
        self.f2=FocalConv2d(64,64,3,parts=4,padding=1)
        self.f3=FocalConv2d(64,128,3,parts=8,padding=1)
        self.f4=FocalConv2d(128,128,3,parts=8,padding=1)
        self.act=nn.LeakyReLU(0.2,inplace=True)
        self.pool=nn.MaxPool2d(2,2)

    def forward(self,x):
        # [B,T,32,64,44] -> [B,T,128,16,11]
        b,t,c,h,w=x.shape
        z=x.reshape(b*t,c,h,w)
        z=self.act(self.f1(z))
        z=self.act(self.f2(z))
        z=self.pool(z)
        z=self.act(self.f3(z))
        z=self.act(self.f4(z))
        z=self.pool(z)
        return z.reshape(b,t,*z.shape[1:])


class _TemporalRelationUnit(nn.Module):
    """One AFRE block; kernel-3 and kernel-5 operations are independent."""
    def __init__(self, channels=128):
        super().__init__()
        self.c3=nn.Conv1d(channels,channels,3,padding=1,bias=True)
        self.c5=nn.Conv1d(channels,channels,5,padding=2,bias=True)

    def _path(self,x,conv,kernel):
        # x [B,C,T]
        cv=conv(x)
        mp=F.max_pool1d(x,kernel,stride=1,padding=kernel//2)
        ap=F.avg_pool1d(x,kernel,stride=1,padding=kernel//2)
        return cv*(mp+ap)

    def forward(self,x):
        # max over temporal dimension after combining two kernel scales
        y=self._path(x,self.c3,3)+self._path(x,self.c5,5)
        return y.max(dim=-1).values  # [B,C]


class AFRE(nn.Module):
    """Adjacent Feature Relation Extractor from RPNet/PFRE."""
    def __init__(self, channels=128, blocks=16, out_channels=256):
        super().__init__()
        self.blocks=blocks
        # Paper states parallel convolution parameters are not shared.
        self.units_a1=nn.ModuleList([_TemporalRelationUnit(channels) for _ in range(blocks)])
        self.units_a3=nn.ModuleList([_TemporalRelationUnit(channels) for _ in range(blocks)])
        self.fc=nn.Linear(channels,out_channels)

    def forward(self,x):
        # x [B,T,C,H,W], expected H=16, W=11
        # mean+max over W -> [B,T,C,H]
        z=x.mean(dim=-1)+x.max(dim=-1).values
        b,t,c,h=z.shape
        if h != self.blocks:
            z=F.adaptive_avg_pool1d(z.permute(0,1,2,3).reshape(b*t,c,h),self.blocks)
            z=z.reshape(b,t,c,self.blocks)
        # A1: each single horizontal body block
        a1=[]
        for i,u in enumerate(self.units_a1):
            xi=z[:,:,:,i].permute(0,2,1).contiguous() # [B,C,T]
            a1.append(u(xi))
        a1=torch.stack(a1,dim=1) # [B,16,C]

        # A3: aggregate each block with its neighbors by direct addition.
        # End blocks are doubled before adding their only neighbor, as described in RPNet.
        a3=[]
        for i,u in enumerate(self.units_a3):
            if i==0:
                xi=2*z[:,:,:,0]+z[:,:,:,1]
            elif i==self.blocks-1:
                xi=z[:,:,:,-2]+2*z[:,:,:,-1]
            else:
                xi=z[:,:,:,i-1]+z[:,:,:,i]+z[:,:,:,i+1]
            xi=xi.permute(0,2,1).contiguous()
            a3.append(u(xi))
        a3=torch.stack(a3,dim=1) # [B,16,C]

        a=torch.cat([a1,a3],dim=1) # [B,32,C]
        a=self.fc(a)               # [B,32,256]
        return a.permute(0,2,1).contiguous() # [B,256,32]


class TPFE(nn.Module):
    """Total-Partial Feature Extractor with n=[1,2,4,8,16]."""
    def __init__(self,channels=128,out_channels=256,bins=(1,2,4,8,16)):
        super().__init__()
        self.bins=bins
        self.fc=nn.Linear(channels,out_channels)

    def forward(self,x):
        # temporal max -> [B,C,H,W]
        z=x.max(dim=1).values
        b,c,h,w=z.shape
        feats=[]
        flat=z.reshape(b,c,h*w)
        for n in self.bins:
            # HPP-equivalent partitioning into n spatial blocks
            if (h*w)%n==0:
                q=flat.reshape(b,c,n,-1)
                f=q.max(dim=-1).values+q.mean(dim=-1)
            else:
                q=F.adaptive_avg_pool1d(flat,n)
                m=F.adaptive_max_pool1d(flat,n)
                f=q+m
            feats.append(f)
        t=torch.cat(feats,dim=-1)       # [B,C,31]
        t=t.permute(0,2,1).contiguous()# [B,31,C]
        t=self.fc(t)                    # [B,31,256]
        return t.permute(0,2,1).contiguous() # [B,256,31]


class FeatureMappingNetwork(nn.Module):
    """PFRE = AFRE + TPFE; final Feature3 has 63 parts."""
    def __init__(self):
        super().__init__()
        self.afre=AFRE()
        self.tpfe=TPFE()

    def forward(self,x):
        a=self.afre(x)
        t=self.tpfe(x)
        return torch.cat([a,t],dim=-1) # [B,256,63]


class DeepGaitGGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generate=GenerateNetwork()
        self.encoder=EncoderNetwork()
        self.mapping=FeatureMappingNetwork()

    def encode_low(self,low):
        high=self.encoder(low)
        emb=self.mapping(high)
        return high,emb

    def forward(self,nm=None,bg=None,cl=None):
        low=self.generate(nm=nm,bg=bg,cl=cl)
        out={}
        for k,v in low.items():
            high,emb=self.encode_low(v)
            out[k]={"low":v,"high":high,"embedding":emb}
        return out

    @staticmethod
    def mapping_loss(outputs,lambda_bg=0.01,lambda_cl=0.1):
        if "nm" not in outputs:
            return torch.tensor(0.0)
        loss=outputs["nm"]["low"].new_tensor(0.0)
        ref=outputs["nm"]["low"]
        if "bg" in outputs:
            loss=loss+lambda_bg*F.l1_loss(outputs["bg"]["low"],ref)
        if "cl" in outputs:
            loss=loss+lambda_cl*F.l1_loss(outputs["cl"]["low"],ref)
        return loss


def flatten_embedding(x):
    return x.flatten(1)


def pairwise_euclidean(query,gallery):
    q=flatten_embedding(query)
    g=flatten_embedding(gallery)
    return torch.cdist(q,g,p=2)
