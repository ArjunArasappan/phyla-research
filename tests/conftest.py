from pathlib import Path
import copy

import pytest
import torch

from flomo.config import Config
from flomo.model import FrozenEncoders
from flomo.sim import collect
from flomo.data import prepare


@pytest.fixture(scope="session",autouse=True)
def cpu_threads(): torch.set_num_threads(1)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp("prepared")
    c=Config(); c.data.raw=str(root/"raw"); c.data.prepared=str(root/"prepared")
    c.eval.horizon=32; c.train.steps=2; c.train.batch_size=2; c.train.save_every=2; c.train.validate_every=2
    collect(c,count=24,human_fraction=.25)
    metadata=prepare(c,FrozenEncoders(c.model))
    return c,metadata


@pytest.fixture
def config(prepared,tmp_path):
    c=copy.deepcopy(prepared[0]); c.train.output=str(tmp_path/"run"); c.eval.output=str(tmp_path/"eval")
    return c
