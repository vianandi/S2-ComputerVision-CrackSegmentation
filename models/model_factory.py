from models.residual_unet import DualAttentionResidualUNet
from models.unet import UNet


def build_model(cfg, device):
    model_cfg = cfg.get("model", {})
    model_name = str(model_cfg.get("name", "dual_attention_residual_unet")).lower()
    in_channels = int(model_cfg.get("in_channels", 3))
    num_classes = int(model_cfg.get("num_classes", 1))
    deep_supervision = bool(model_cfg.get("deep_supervision", False))
    edge_supervision = bool(model_cfg.get("edge_supervision", False))
    skip_attention = bool(model_cfg.get("skip_attention", False))
    aspp_dilations = model_cfg.get("aspp_dilations")
    aspp_use_global_pool = bool(model_cfg.get("aspp_use_global_pool", True))
    aspp_use_strip_pool = bool(model_cfg.get("aspp_use_strip_pool", False))
    aspp_dropout = float(model_cfg.get("aspp_dropout", 0.5))

    if model_name == "unet":
        model = UNet(in_channels=in_channels, num_classes=num_classes)
    elif model_name in {"dual_attention_residual_unet", "darunet", "proposed"}:
        base_channels = int(model_cfg.get("base_channels", 64))
        model = DualAttentionResidualUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision,
            edge_supervision=edge_supervision,
            skip_attention=skip_attention,
            aspp_dilations=aspp_dilations,
            aspp_use_global_pool=aspp_use_global_pool,
            aspp_use_strip_pool=aspp_use_strip_pool,
            aspp_dropout=aspp_dropout,
        )
    else:
        valid = ["unet", "dual_attention_residual_unet"]
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")

    return model.to(device), model_name
