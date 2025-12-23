# tbm/tools.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Union
import joblib
from tqdm import tqdm


class Tool(ABC):
    """
    Contract:
      - process(image_paths, save_dir, **kwargs) -> List[Dict]
        Each dict MUST contain:
          {"tool": <str>, "input_path": <str>, "output_path": <str>, "schema": <str>}
      - get_output_info() -> metadata (type, schema, etc.)
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
    
    @abstractmethod
    def process(self, image_paths: List, save_dir: str, **kwargs) -> List[Path]:
        """Process images and save outputs."""
        pass
    
    @abstractmethod
    def get_output_info(self) -> Dict[str, Any]:
        """Return metadata about outputs."""
        pass


class HoverNetTool(Tool):
    """HoVer-Net nucleus segmentation."""
    
    def __init__(self, name="hovernet", model="hovernet_fast-pannuke", 
                 device="cuda", batch_size=16, **kwargs):
        super().__init__(name, **kwargs)
        from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
        
        self.segmentor = NucleusInstanceSegmentor(
            pretrained_model=model,
            batch_size=batch_size,
            auto_generate_mask=False,
            verbose=False
        )
        self.device = device

    def process(self, image_paths: List, save_dir: str, **kwargs) -> List[dict]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = "all_hovernet_outputs"
        outputs = self.segmentor.predict(
            [str(p) for p in image_paths],
            mode="tile",
            device=self.device,
            save_dir=str(temp_dir),
            crash_on_exception=True,
            # if you expose these in __init__, forward them:
            # num_loader_workers=self.num_loader_workers,
            # num_postproc_workers=self.num_postproc_workers,
        )

        results = []
        for inp_path, out_base in tqdm(outputs, desc="Saving"):
            dat_path = str(out_base) if str(out_base).endswith(".dat") else str(out_base) + ".dat"
            inst_dict = joblib.load(dat_path)

            stem = Path(inp_path).stem
            final_path = save_dir / f"{stem}_{self.name}.dat"
            joblib.dump(inst_dict, final_path)
            results.append({"input_path": Path(inp_path), "output_path": final_path})

            Path(dat_path).unlink(missing_ok=True)

        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return results
    
    def get_output_info(self) -> Dict[str, Any]:
        return {
            'output_type': 'nucleus_instances',
            'file_extension': '.dat',
            'num_types': 6,
            'type_names': ['background', 'neoplastic_epithelial', 'inflammatory',
                          'connective', 'dead', 'non_neoplastic_epithelial']
        }

# Users add their tools here:
# class MyCustomTool(Tool):
#     def process(self, image_paths, save_dir, **kwargs):
#         # Your implementation
#         pass