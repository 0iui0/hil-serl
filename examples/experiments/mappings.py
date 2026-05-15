from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.usb_pickup_insertion.config import TrainConfig as USBPickupInsertionTrainConfig
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.motor_shaft_assembly.cr5af.config import TrainConfig as MotorShaftCR5AFTrainConfig
from experiments.motor_shaft_assembly.marvin.config import TrainConfig as MotorShaftMarvinTrainConfig

CONFIG_MAPPING = {
    "ram_insertion": RAMInsertionTrainConfig,
    "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    "object_handover": ObjectHandoverTrainConfig,
    "egg_flip": EggFlipTrainConfig,
    "motor_shaft_assembly_cr5af": MotorShaftCR5AFTrainConfig,
    "motor_shaft_assembly_marvin": MotorShaftMarvinTrainConfig,
}