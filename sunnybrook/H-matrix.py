#
# Copyright 2016 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Do NOT execute this file unless you have the SunnyBrook data in a subdir!

import caffe
import numpy as np
import h5py

# [cost of predicting 1 when gt is 0,    cost of predicting 0 when gt is 0
#  cost of predicting 1 when gt is 1,    cost of predicting 0 when gt is 1]
H = np.array([[0.02, 0], [0, 0.98]], dtype='f4')

import caffe
blob = caffe.io.array_to_blobproto( H.reshape( (1,1,2,2) ) )
with open( 'infogainH.binaryproto', 'wb' ) as f :
    f.write( blob.SerializeToString() )

db = h5py.File('infoGainH.hdf5', 'w')
db.create_dataset('infogainWeights', data = H)



