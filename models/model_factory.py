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

    residual_variants = {
        "residual_unet": (False, False, "residual_unet"),
        "unet_residual": (False, False, "residual_unet"),
        "residual_attention_unet": (True, False, "residual_attention_unet"),
        "unet_residual_attention": (True, False, "residual_attention_unet"),
        "dual_attention_residual_unet": (True, True, "dual_attention_residual_unet"),
        "darunet": (True, True, "dual_attention_residual_unet"),
        "proposed": (True, True, "dual_attention_residual_unet"),
    }

    if model_name == "unet":
        model = UNet(in_channels=in_channels, num_classes=num_classes)
    elif model_name in residual_variants:
        use_attention, use_aspp, canonical_name = residual_variants[model_name]
        base_channels = int(model_cfg.get("base_channels", 64))
        model = DualAttentionResidualUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=deep_supervision and use_aspp,
            edge_supervision=edge_supervision and use_aspp,
            skip_attention=skip_attention and use_attention,
            aspp_dilations=aspp_dilations,
            aspp_use_global_pool=aspp_use_global_pool,
            aspp_use_strip_pool=aspp_use_strip_pool,
            aspp_dropout=aspp_dropout,
            use_attention=use_attention,
            use_aspp=use_aspp,
        )
        model_name = canonical_name
    else:
        valid = [
            "unet",
            "residual_unet",
            "residual_attention_unet",
            "dual_attention_residual_unet",
        ]
        raise ValueError(f"Unknown model '{model_name}'. Valid options: {valid}")

    return model.to(device), model_name
