�
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.0.02unknown8��
�
baselayer_2076/kernelVarHandleOp*
shape
:2*&
shared_namebaselayer_2076/kernel*
dtype0*
_output_shapes
: 

)baselayer_2076/kernel/Read/ReadVariableOpReadVariableOpbaselayer_2076/kernel*
dtype0*
_output_shapes

:2
~
baselayer_2076/biasVarHandleOp*
shape:2*$
shared_namebaselayer_2076/bias*
dtype0*
_output_shapes
: 
w
'baselayer_2076/bias/Read/ReadVariableOpReadVariableOpbaselayer_2076/bias*
dtype0*
_output_shapes
:2
�
dense_var_layer_2076/kernelVarHandleOp*
shape
:2*,
shared_namedense_var_layer_2076/kernel*
dtype0*
_output_shapes
: 
�
/dense_var_layer_2076/kernel/Read/ReadVariableOpReadVariableOpdense_var_layer_2076/kernel*
dtype0*
_output_shapes

:2
�
dense_var_layer_2076/biasVarHandleOp*
shape:**
shared_namedense_var_layer_2076/bias*
dtype0*
_output_shapes
: 
�
-dense_var_layer_2076/bias/Read/ReadVariableOpReadVariableOpdense_var_layer_2076/bias*
dtype0*
_output_shapes
:
�
pi_layer_2076/kernelVarHandleOp*
shape
:2*%
shared_namepi_layer_2076/kernel*
dtype0*
_output_shapes
: 
}
(pi_layer_2076/kernel/Read/ReadVariableOpReadVariableOppi_layer_2076/kernel*
dtype0*
_output_shapes

:2
|
pi_layer_2076/biasVarHandleOp*
shape:*#
shared_namepi_layer_2076/bias*
dtype0*
_output_shapes
: 
u
&pi_layer_2076/bias/Read/ReadVariableOpReadVariableOppi_layer_2076/bias*
dtype0*
_output_shapes
:
�
mean_layer_2076/kernelVarHandleOp*
shape
:2*'
shared_namemean_layer_2076/kernel*
dtype0*
_output_shapes
: 
�
*mean_layer_2076/kernel/Read/ReadVariableOpReadVariableOpmean_layer_2076/kernel*
dtype0*
_output_shapes

:2
�
mean_layer_2076/biasVarHandleOp*
shape:*%
shared_namemean_layer_2076/bias*
dtype0*
_output_shapes
: 
y
(mean_layer_2076/bias/Read/ReadVariableOpReadVariableOpmean_layer_2076/bias*
dtype0*
_output_shapes
:

NoOpNoOp
�
ConstConst"/device:CPU:0*�
value�B� B�
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
regularization_losses
trainable_variables
		variables

	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
�
(	arguments
)_variable_dict
*_trainable_weights
+_non_trainable_weights
,regularization_losses
-trainable_variables
.	variables
/	keras_api
 
8
0
1
2
3
4
5
"6
#7
8
0
1
2
3
4
5
"6
#7
�
regularization_losses

0layers
1layer_regularization_losses
trainable_variables
2metrics
3non_trainable_variables
		variables
 
 
 
 
�
regularization_losses
4layer_regularization_losses

5layers
trainable_variables
6metrics
7non_trainable_variables
	variables
a_
VARIABLE_VALUEbaselayer_2076/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEbaselayer_2076/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
8layer_regularization_losses

9layers
trainable_variables
:metrics
;non_trainable_variables
	variables
ge
VARIABLE_VALUEdense_var_layer_2076/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEdense_var_layer_2076/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
<layer_regularization_losses

=layers
trainable_variables
>metrics
?non_trainable_variables
	variables
`^
VARIABLE_VALUEpi_layer_2076/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEpi_layer_2076/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
@layer_regularization_losses

Alayers
trainable_variables
Bmetrics
Cnon_trainable_variables
 	variables
b`
VARIABLE_VALUEmean_layer_2076/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEmean_layer_2076/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
�
$regularization_losses
Dlayer_regularization_losses

Elayers
%trainable_variables
Fmetrics
Gnon_trainable_variables
&	variables
 
 
 
 
 
 
 
�
,regularization_losses
Hlayer_regularization_losses

Ilayers
-trainable_variables
Jmetrics
Knon_trainable_variables
.	variables
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
z
serving_default_input_5Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5baselayer_2076/kernelbaselayer_2076/biasdense_var_layer_2076/kerneldense_var_layer_2076/biasmean_layer_2076/kernelmean_layer_2076/biaspi_layer_2076/kernelpi_layer_2076/bias*/
_gradient_op_typePartitionedCall-65910731*/
f*R(
&__inference_signature_wrapper_65910513*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:���������
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)baselayer_2076/kernel/Read/ReadVariableOp'baselayer_2076/bias/Read/ReadVariableOp/dense_var_layer_2076/kernel/Read/ReadVariableOp-dense_var_layer_2076/bias/Read/ReadVariableOp(pi_layer_2076/kernel/Read/ReadVariableOp&pi_layer_2076/bias/Read/ReadVariableOp*mean_layer_2076/kernel/Read/ReadVariableOp(mean_layer_2076/bias/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-65910763**
f%R#
!__inference__traced_save_65910762*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2
*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebaselayer_2076/kernelbaselayer_2076/biasdense_var_layer_2076/kerneldense_var_layer_2076/biaspi_layer_2076/kernelpi_layer_2076/biasmean_layer_2076/kernelmean_layer_2076/bias*/
_gradient_op_typePartitionedCall-65910800*-
f(R&
$__inference__traced_restore_65910799*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*
_output_shapes
: ާ
�#
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910478

inputs,
(baselayer_statefulpartitionedcall_args_1,
(baselayer_statefulpartitionedcall_args_22
.dense_var_layer_statefulpartitionedcall_args_12
.dense_var_layer_statefulpartitionedcall_args_2-
)mean_layer_statefulpartitionedcall_args_1-
)mean_layer_statefulpartitionedcall_args_2+
'pi_layer_statefulpartitionedcall_args_1+
'pi_layer_statefulpartitionedcall_args_2
identity

identity_1

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910260*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������2�
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910288*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:����������
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-65910324*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910312*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910345*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910373*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
�#
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910411
input_5,
(baselayer_statefulpartitionedcall_args_1,
(baselayer_statefulpartitionedcall_args_22
.dense_var_layer_statefulpartitionedcall_args_12
.dense_var_layer_statefulpartitionedcall_args_2-
)mean_layer_statefulpartitionedcall_args_1-
)mean_layer_statefulpartitionedcall_args_2+
'pi_layer_statefulpartitionedcall_args_1+
'pi_layer_statefulpartitionedcall_args_2
identity

identity_1

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall`
baselayer/CastCastinput_5*

SrcT0*

DstT0*'
_output_shapes
:����������
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910260*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������2�
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910288*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:����������
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-65910324*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910312*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910345*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910373*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_5: : : 
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910305

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:���������O
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
-__inference_mean_layer_layer_call_fn_65910691

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910345*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�
�
&__inference_signature_wrapper_65910513
input_5"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-65910498*,
f'R%
#__inference__wrapped_model_65910236*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_5: : : 
�	
�
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�0
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910587

inputs,
(baselayer_matmul_readvariableop_resource-
)baselayer_biasadd_readvariableop_resource2
.dense_var_layer_matmul_readvariableop_resource3
/dense_var_layer_biasadd_readvariableop_resource-
)mean_layer_matmul_readvariableop_resource.
*mean_layer_biasadd_readvariableop_resource+
'pi_layer_matmul_readvariableop_resource,
(pi_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2�� baselayer/BiasAdd/ReadVariableOp�baselayer/MatMul/ReadVariableOp�&dense_var_layer/BiasAdd/ReadVariableOp�%dense_var_layer/MatMul/ReadVariableOp�!mean_layer/BiasAdd/ReadVariableOp� mean_layer/MatMul/ReadVariableOp�pi_layer/BiasAdd/ReadVariableOp�pi_layer/MatMul/ReadVariableOp_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
baselayer/MatMulMatMulbaselayer/Cast:y:0'baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 baselayer/BiasAdd/ReadVariableOpReadVariableOp)baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2�
baselayer/BiasAddBiasAddbaselayer/MatMul:product:0(baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
baselayer/TanhTanhbaselayer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2q
dense_var_layer/CastCastbaselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
dense_var_layer/MatMulMatMuldense_var_layer/Cast:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
variance_layer/CastCast dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������d
variance_layer/ExpExpvariance_layer/Cast:y:0*
T0*'
_output_shapes
:����������
 mean_layer/MatMul/ReadVariableOpReadVariableOp)mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
mean_layer/MatMulMatMulbaselayer/Tanh:y:0(mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!mean_layer/BiasAdd/ReadVariableOpReadVariableOp*mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
mean_layer/BiasAddBiasAddmean_layer/MatMul:product:0)mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
pi_layer/MatMul/ReadVariableOpReadVariableOp'pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
pi_layer/MatMulMatMulbaselayer/Tanh:y:0&pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
pi_layer/BiasAdd/ReadVariableOpReadVariableOp(pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
pi_layer/BiasAddBiasAddpi_layer/MatMul:product:0'pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
pi_layer/SoftmaxSoftmaxpi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitypi_layer/Softmax:softmax:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identitymean_layer/BiasAdd:output:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_2Identityvariance_layer/Exp:y:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2@
pi_layer/MatMul/ReadVariableOppi_layer/MatMul/ReadVariableOp2D
 mean_layer/MatMul/ReadVariableOp mean_layer/MatMul/ReadVariableOp2D
 baselayer/BiasAdd/ReadVariableOp baselayer/BiasAdd/ReadVariableOp2B
pi_layer/BiasAdd/ReadVariableOppi_layer/BiasAdd/ReadVariableOp2F
!mean_layer/BiasAdd/ReadVariableOp!mean_layer/BiasAdd/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
�	
�
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
!__inference__traced_save_65910762
file_prefix4
0savev2_baselayer_2076_kernel_read_readvariableop2
.savev2_baselayer_2076_bias_read_readvariableop:
6savev2_dense_var_layer_2076_kernel_read_readvariableop8
4savev2_dense_var_layer_2076_bias_read_readvariableop3
/savev2_pi_layer_2076_kernel_read_readvariableop1
-savev2_pi_layer_2076_bias_read_readvariableop5
1savev2_mean_layer_2076_kernel_read_readvariableop3
/savev2_mean_layer_2076_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_a90bf0c7de274f1b8358d4013974b9ac/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:}
SaveV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_baselayer_2076_kernel_read_readvariableop.savev2_baselayer_2076_bias_read_readvariableop6savev2_dense_var_layer_2076_kernel_read_readvariableop4savev2_dense_var_layer_2076_bias_read_readvariableop/savev2_pi_layer_2076_kernel_read_readvariableop-savev2_pi_layer_2076_bias_read_readvariableop1savev2_mean_layer_2076_kernel_read_readvariableop/savev2_mean_layer_2076_bias_read_readvariableop"/device:CPU:0*
dtypes

2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: �
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 �
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*W
_input_shapesF
D: :2:2:2::2::2:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :+ '
%
_user_specified_namefile_prefix: : :	 : 
�	
�
G__inference_baselayer_layer_call_and_return_conditional_losses_65910632

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*.
_input_shapes
:���������::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910667

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
*__inference_model_4_layer_call_fn_65910604

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-65910437*N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_65910436*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
�
�
*__inference_model_4_layer_call_fn_65910452
input_5"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-65910437*N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_65910436*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_5: : : 
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910696

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:���������O
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910312

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:���������O
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������2�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
*__inference_model_4_layer_call_fn_65910621

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-65910479*N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_65910478*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
�	
�
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910684

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
M
1__inference_variance_layer_layer_call_fn_65910711

inputs
identity�
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-65910324*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910312*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910649

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910701

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:���������O
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�	
�
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
2__inference_dense_var_layer_layer_call_fn_65910656

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910288*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�#
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910387
input_5,
(baselayer_statefulpartitionedcall_args_1,
(baselayer_statefulpartitionedcall_args_22
.dense_var_layer_statefulpartitionedcall_args_12
.dense_var_layer_statefulpartitionedcall_args_2-
)mean_layer_statefulpartitionedcall_args_1-
)mean_layer_statefulpartitionedcall_args_2+
'pi_layer_statefulpartitionedcall_args_1+
'pi_layer_statefulpartitionedcall_args_2
identity

identity_1

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall`
baselayer/CastCastinput_5*

SrcT0*

DstT0*'
_output_shapes
:����������
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910260*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������2�
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910288*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:����������
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-65910316*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910305*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910345*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910373*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_5: : : 
�#
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910436

inputs,
(baselayer_statefulpartitionedcall_args_1,
(baselayer_statefulpartitionedcall_args_22
.dense_var_layer_statefulpartitionedcall_args_12
.dense_var_layer_statefulpartitionedcall_args_2-
)mean_layer_statefulpartitionedcall_args_1-
)mean_layer_statefulpartitionedcall_args_2+
'pi_layer_statefulpartitionedcall_args_1+
'pi_layer_statefulpartitionedcall_args_2
identity

identity_1

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910260*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������2�
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910288*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910282*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:����������
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-65910316*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910305*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910345*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910339*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910373*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
�
�
+__inference_pi_layer_layer_call_fn_65910674

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910373*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910367*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*.
_input_shapes
:���������2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�6
�
#__inference__wrapped_model_65910236
input_54
0model_4_baselayer_matmul_readvariableop_resource5
1model_4_baselayer_biasadd_readvariableop_resource:
6model_4_dense_var_layer_matmul_readvariableop_resource;
7model_4_dense_var_layer_biasadd_readvariableop_resource5
1model_4_mean_layer_matmul_readvariableop_resource6
2model_4_mean_layer_biasadd_readvariableop_resource3
/model_4_pi_layer_matmul_readvariableop_resource4
0model_4_pi_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2��(model_4/baselayer/BiasAdd/ReadVariableOp�'model_4/baselayer/MatMul/ReadVariableOp�.model_4/dense_var_layer/BiasAdd/ReadVariableOp�-model_4/dense_var_layer/MatMul/ReadVariableOp�)model_4/mean_layer/BiasAdd/ReadVariableOp�(model_4/mean_layer/MatMul/ReadVariableOp�'model_4/pi_layer/BiasAdd/ReadVariableOp�&model_4/pi_layer/MatMul/ReadVariableOph
model_4/baselayer/CastCastinput_5*

SrcT0*

DstT0*'
_output_shapes
:����������
'model_4/baselayer/MatMul/ReadVariableOpReadVariableOp0model_4_baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model_4/baselayer/MatMulMatMulmodel_4/baselayer/Cast:y:0/model_4/baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
(model_4/baselayer/BiasAdd/ReadVariableOpReadVariableOp1model_4_baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2�
model_4/baselayer/BiasAddBiasAdd"model_4/baselayer/MatMul:product:00model_4/baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2t
model_4/baselayer/TanhTanh"model_4/baselayer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
model_4/dense_var_layer/CastCastmodel_4/baselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
-model_4/dense_var_layer/MatMul/ReadVariableOpReadVariableOp6model_4_dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model_4/dense_var_layer/MatMulMatMul model_4/dense_var_layer/Cast:y:05model_4/dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
.model_4/dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp7model_4_dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model_4/dense_var_layer/BiasAddBiasAdd(model_4/dense_var_layer/MatMul:product:06model_4/dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model_4/variance_layer/CastCast(model_4/dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������t
model_4/variance_layer/ExpExpmodel_4/variance_layer/Cast:y:0*
T0*'
_output_shapes
:����������
(model_4/mean_layer/MatMul/ReadVariableOpReadVariableOp1model_4_mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model_4/mean_layer/MatMulMatMulmodel_4/baselayer/Tanh:y:00model_4/mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
)model_4/mean_layer/BiasAdd/ReadVariableOpReadVariableOp2model_4_mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model_4/mean_layer/BiasAddBiasAdd#model_4/mean_layer/MatMul:product:01model_4/mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_4/pi_layer/MatMul/ReadVariableOpReadVariableOp/model_4_pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model_4/pi_layer/MatMulMatMulmodel_4/baselayer/Tanh:y:0.model_4/pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model_4/pi_layer/BiasAdd/ReadVariableOpReadVariableOp0model_4_pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model_4/pi_layer/BiasAddBiasAdd!model_4/pi_layer/MatMul:product:0/model_4/pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
model_4/pi_layer/SoftmaxSoftmax!model_4/pi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity#model_4/mean_layer/BiasAdd:output:0)^model_4/baselayer/BiasAdd/ReadVariableOp(^model_4/baselayer/MatMul/ReadVariableOp/^model_4/dense_var_layer/BiasAdd/ReadVariableOp.^model_4/dense_var_layer/MatMul/ReadVariableOp*^model_4/mean_layer/BiasAdd/ReadVariableOp)^model_4/mean_layer/MatMul/ReadVariableOp(^model_4/pi_layer/BiasAdd/ReadVariableOp'^model_4/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identity"model_4/pi_layer/Softmax:softmax:0)^model_4/baselayer/BiasAdd/ReadVariableOp(^model_4/baselayer/MatMul/ReadVariableOp/^model_4/dense_var_layer/BiasAdd/ReadVariableOp.^model_4/dense_var_layer/MatMul/ReadVariableOp*^model_4/mean_layer/BiasAdd/ReadVariableOp)^model_4/mean_layer/MatMul/ReadVariableOp(^model_4/pi_layer/BiasAdd/ReadVariableOp'^model_4/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_2Identitymodel_4/variance_layer/Exp:y:0)^model_4/baselayer/BiasAdd/ReadVariableOp(^model_4/baselayer/MatMul/ReadVariableOp/^model_4/dense_var_layer/BiasAdd/ReadVariableOp.^model_4/dense_var_layer/MatMul/ReadVariableOp*^model_4/mean_layer/BiasAdd/ReadVariableOp)^model_4/mean_layer/MatMul/ReadVariableOp(^model_4/pi_layer/BiasAdd/ReadVariableOp'^model_4/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2T
(model_4/baselayer/BiasAdd/ReadVariableOp(model_4/baselayer/BiasAdd/ReadVariableOp2R
'model_4/baselayer/MatMul/ReadVariableOp'model_4/baselayer/MatMul/ReadVariableOp2V
)model_4/mean_layer/BiasAdd/ReadVariableOp)model_4/mean_layer/BiasAdd/ReadVariableOp2`
.model_4/dense_var_layer/BiasAdd/ReadVariableOp.model_4/dense_var_layer/BiasAdd/ReadVariableOp2P
&model_4/pi_layer/MatMul/ReadVariableOp&model_4/pi_layer/MatMul/ReadVariableOp2^
-model_4/dense_var_layer/MatMul/ReadVariableOp-model_4/dense_var_layer/MatMul/ReadVariableOp2R
'model_4/pi_layer/BiasAdd/ReadVariableOp'model_4/pi_layer/BiasAdd/ReadVariableOp2T
(model_4/mean_layer/MatMul/ReadVariableOp(model_4/mean_layer/MatMul/ReadVariableOp: : : : : :' #
!
_user_specified_name	input_5: : : 
�&
�
$__inference__traced_restore_65910799
file_prefix*
&assignvariableop_baselayer_2076_kernel*
&assignvariableop_1_baselayer_2076_bias2
.assignvariableop_2_dense_var_layer_2076_kernel0
,assignvariableop_3_dense_var_layer_2076_bias+
'assignvariableop_4_pi_layer_2076_kernel)
%assignvariableop_5_pi_layer_2076_bias-
)assignvariableop_6_mean_layer_2076_kernel+
'assignvariableop_7_mean_layer_2076_bias

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:�
RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes

2*4
_output_shapes"
 ::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_baselayer_2076_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_baselayer_2076_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp.assignvariableop_2_dense_var_layer_2076_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_dense_var_layer_2076_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp'assignvariableop_4_pi_layer_2076_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp%assignvariableop_5_pi_layer_2076_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp)assignvariableop_6_mean_layer_2076_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_mean_layer_2076_biasIdentity_7:output:0*
dtype0*
_output_shapes
 �
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: �

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2(
AssignVariableOp_7AssignVariableOp_72
RestoreV2_1RestoreV2_12
	RestoreV2	RestoreV22(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6: : : : : :+ '
%
_user_specified_namefile_prefix: : : 
�
�
*__inference_model_4_layer_call_fn_65910494
input_5"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity

identity_1

identity_2��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_5statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-65910479*N
fIRG
E__inference_model_4_layer_call_and_return_conditional_losses_65910478*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:����������
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:����������

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_5: : : 
�
�
,__inference_baselayer_layer_call_fn_65910639

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-65910260*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_65910254*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������2�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
�0
�
E__inference_model_4_layer_call_and_return_conditional_losses_65910551

inputs,
(baselayer_matmul_readvariableop_resource-
)baselayer_biasadd_readvariableop_resource2
.dense_var_layer_matmul_readvariableop_resource3
/dense_var_layer_biasadd_readvariableop_resource-
)mean_layer_matmul_readvariableop_resource.
*mean_layer_biasadd_readvariableop_resource+
'pi_layer_matmul_readvariableop_resource,
(pi_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2�� baselayer/BiasAdd/ReadVariableOp�baselayer/MatMul/ReadVariableOp�&dense_var_layer/BiasAdd/ReadVariableOp�%dense_var_layer/MatMul/ReadVariableOp�!mean_layer/BiasAdd/ReadVariableOp� mean_layer/MatMul/ReadVariableOp�pi_layer/BiasAdd/ReadVariableOp�pi_layer/MatMul/ReadVariableOp_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:����������
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
baselayer/MatMulMatMulbaselayer/Cast:y:0'baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
 baselayer/BiasAdd/ReadVariableOpReadVariableOp)baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2�
baselayer/BiasAddBiasAddbaselayer/MatMul:product:0(baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2d
baselayer/TanhTanhbaselayer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2q
dense_var_layer/CastCastbaselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:���������2�
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
dense_var_layer/MatMulMatMuldense_var_layer/Cast:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
variance_layer/CastCast dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:���������d
variance_layer/ExpExpvariance_layer/Cast:y:0*
T0*'
_output_shapes
:����������
 mean_layer/MatMul/ReadVariableOpReadVariableOp)mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
mean_layer/MatMulMatMulbaselayer/Tanh:y:0(mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!mean_layer/BiasAdd/ReadVariableOpReadVariableOp*mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
mean_layer/BiasAddBiasAddmean_layer/MatMul:product:0)mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
pi_layer/MatMul/ReadVariableOpReadVariableOp'pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
pi_layer/MatMulMatMulbaselayer/Tanh:y:0&pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
pi_layer/BiasAdd/ReadVariableOpReadVariableOp(pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
pi_layer/BiasAddBiasAddpi_layer/MatMul:product:0'pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
pi_layer/SoftmaxSoftmaxpi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentitypi_layer/Softmax:softmax:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identitymean_layer/BiasAdd:output:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_2Identityvariance_layer/Exp:y:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2D
 mean_layer/MatMul/ReadVariableOp mean_layer/MatMul/ReadVariableOp2@
pi_layer/MatMul/ReadVariableOppi_layer/MatMul/ReadVariableOp2D
 baselayer/BiasAdd/ReadVariableOp baselayer/BiasAdd/ReadVariableOp2F
!mean_layer/BiasAdd/ReadVariableOp!mean_layer/BiasAdd/ReadVariableOp2B
pi_layer/BiasAdd/ReadVariableOppi_layer/BiasAdd/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
�
M
1__inference_variance_layer_layer_call_fn_65910706

inputs
identity�
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-65910316*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910305*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2*'
_output_shapes
:���������`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*&
_input_shapes
:���������:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
;
input_50
serving_default_input_5:0���������<
pi_layer0
StatefulPartitionedCall:1���������>

mean_layer0
StatefulPartitionedCall:0���������B
variance_layer0
StatefulPartitionedCall:2���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�1
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
regularization_losses
trainable_variables
		variables

	keras_api

signatures
L__call__
*M&call_and_return_all_conditional_losses
N_default_save_signature"�.
_tf_keras_model�.{"class_name": "Model", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT79AAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT79AAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}}}
�
regularization_losses
trainable_variables
	variables
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "input_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 5], "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_5"}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "baselayer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_var_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
U__call__
*V&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "pi_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
W__call__
*X&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "mean_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�
(	arguments
)_variable_dict
*_trainable_weights
+_non_trainable_weights
,regularization_losses
-trainable_variables
.	variables
/	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Lambda", "name": "variance_layer", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT79AAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
 "
trackable_list_wrapper
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
X
0
1
2
3
4
5
"6
#7"
trackable_list_wrapper
�
regularization_losses

0layers
1layer_regularization_losses
trainable_variables
2metrics
3non_trainable_variables
		variables
L__call__
N_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
[serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
regularization_losses
4layer_regularization_losses

5layers
trainable_variables
6metrics
7non_trainable_variables
	variables
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
':%22baselayer_2076/kernel
!:22baselayer_2076/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
8layer_regularization_losses

9layers
trainable_variables
:metrics
;non_trainable_variables
	variables
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
-:+22dense_var_layer_2076/kernel
':%2dense_var_layer_2076/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
<layer_regularization_losses

=layers
trainable_variables
>metrics
?non_trainable_variables
	variables
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
&:$22pi_layer_2076/kernel
 :2pi_layer_2076/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
@layer_regularization_losses

Alayers
trainable_variables
Bmetrics
Cnon_trainable_variables
 	variables
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
(:&22mean_layer_2076/kernel
": 2mean_layer_2076/bias
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
�
$regularization_losses
Dlayer_regularization_losses

Elayers
%trainable_variables
Fmetrics
Gnon_trainable_variables
&	variables
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
,regularization_losses
Hlayer_regularization_losses

Ilayers
-trainable_variables
Jmetrics
Knon_trainable_variables
.	variables
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
*__inference_model_4_layer_call_fn_65910494
*__inference_model_4_layer_call_fn_65910621
*__inference_model_4_layer_call_fn_65910604
*__inference_model_4_layer_call_fn_65910452�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_model_4_layer_call_and_return_conditional_losses_65910551
E__inference_model_4_layer_call_and_return_conditional_losses_65910587
E__inference_model_4_layer_call_and_return_conditional_losses_65910387
E__inference_model_4_layer_call_and_return_conditional_losses_65910411�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference__wrapped_model_65910236�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_5���������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
,__inference_baselayer_layer_call_fn_65910639�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_baselayer_layer_call_and_return_conditional_losses_65910632�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
2__inference_dense_var_layer_layer_call_fn_65910656�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910649�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
+__inference_pi_layer_layer_call_fn_65910674�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910667�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
-__inference_mean_layer_layer_call_fn_65910691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910684�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_variance_layer_layer_call_fn_65910706
1__inference_variance_layer_layer_call_fn_65910711�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910696
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910701�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
5B3
&__inference_signature_wrapper_65910513input_5�
*__inference_model_4_layer_call_fn_65910604�"#7�4
-�*
 �
inputs���������
p

 
� "Z�W
�
0���������
�
1���������
�
2���������~
+__inference_pi_layer_layer_call_fn_65910674O/�,
%�"
 �
inputs���������2
� "�����������
G__inference_baselayer_layer_call_and_return_conditional_losses_65910632\/�,
%�"
 �
inputs���������
� "%�"
�
0���������2
� �
&__inference_signature_wrapper_65910513�"#;�8
� 
1�.
,
input_5!�
input_5���������"���
.
pi_layer"�
pi_layer���������
2

mean_layer$�!

mean_layer���������
:
variance_layer(�%
variance_layer����������
*__inference_model_4_layer_call_fn_65910452�"#8�5
.�+
!�
input_5���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
*__inference_model_4_layer_call_fn_65910621�"#7�4
-�*
 �
inputs���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2����������
1__inference_variance_layer_layer_call_fn_65910711S7�4
-�*
 �
inputs���������

 
p 
� "�����������
1__inference_variance_layer_layer_call_fn_65910706S7�4
-�*
 �
inputs���������

 
p
� "�����������
E__inference_model_4_layer_call_and_return_conditional_losses_65910551�"#7�4
-�*
 �
inputs���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
E__inference_model_4_layer_call_and_return_conditional_losses_65910387�"#8�5
.�+
!�
input_5���������
p

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
H__inference_mean_layer_layer_call_and_return_conditional_losses_65910684\"#/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910696`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������
� �
F__inference_pi_layer_layer_call_and_return_conditional_losses_65910667\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
-__inference_mean_layer_layer_call_fn_65910691O"#/�,
%�"
 �
inputs���������2
� "�����������
*__inference_model_4_layer_call_fn_65910494�"#8�5
.�+
!�
input_5���������
p 

 
� "Z�W
�
0���������
�
1���������
�
2���������
,__inference_baselayer_layer_call_fn_65910639O/�,
%�"
 �
inputs���������
� "����������2�
L__inference_variance_layer_layer_call_and_return_conditional_losses_65910701`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� �
#__inference__wrapped_model_65910236�"#0�-
&�#
!�
input_5���������
� "���
.
pi_layer"�
pi_layer���������
2

mean_layer$�!

mean_layer���������
:
variance_layer(�%
variance_layer����������
E__inference_model_4_layer_call_and_return_conditional_losses_65910411�"#8�5
.�+
!�
input_5���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� �
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_65910649\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
2__inference_dense_var_layer_layer_call_fn_65910656O/�,
%�"
 �
inputs���������2
� "�����������
E__inference_model_4_layer_call_and_return_conditional_losses_65910587�"#7�4
-�*
 �
inputs���������
p 

 
� "j�g
`�]
�
0/0���������
�
0/1���������
�
0/2���������
� 