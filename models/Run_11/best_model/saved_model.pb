ً
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
baselayer_2/kernelVarHandleOp*
shape
:2*#
shared_namebaselayer_2/kernel*
dtype0*
_output_shapes
: 
y
&baselayer_2/kernel/Read/ReadVariableOpReadVariableOpbaselayer_2/kernel*
dtype0*
_output_shapes

:2
x
baselayer_2/biasVarHandleOp*
shape:2*!
shared_namebaselayer_2/bias*
dtype0*
_output_shapes
: 
q
$baselayer_2/bias/Read/ReadVariableOpReadVariableOpbaselayer_2/bias*
dtype0*
_output_shapes
:2
�
dense_var_layer_2/kernelVarHandleOp*
shape
:2*)
shared_namedense_var_layer_2/kernel*
dtype0*
_output_shapes
: 
�
,dense_var_layer_2/kernel/Read/ReadVariableOpReadVariableOpdense_var_layer_2/kernel*
dtype0*
_output_shapes

:2
�
dense_var_layer_2/biasVarHandleOp*
shape:*'
shared_namedense_var_layer_2/bias*
dtype0*
_output_shapes
: 
}
*dense_var_layer_2/bias/Read/ReadVariableOpReadVariableOpdense_var_layer_2/bias*
dtype0*
_output_shapes
:
~
pi_layer_2/kernelVarHandleOp*
shape
:2*"
shared_namepi_layer_2/kernel*
dtype0*
_output_shapes
: 
w
%pi_layer_2/kernel/Read/ReadVariableOpReadVariableOppi_layer_2/kernel*
dtype0*
_output_shapes

:2
v
pi_layer_2/biasVarHandleOp*
shape:* 
shared_namepi_layer_2/bias*
dtype0*
_output_shapes
: 
o
#pi_layer_2/bias/Read/ReadVariableOpReadVariableOppi_layer_2/bias*
dtype0*
_output_shapes
:
�
mean_layer_2/kernelVarHandleOp*
shape
:2*$
shared_namemean_layer_2/kernel*
dtype0*
_output_shapes
: 
{
'mean_layer_2/kernel/Read/ReadVariableOpReadVariableOpmean_layer_2/kernel*
dtype0*
_output_shapes

:2
z
mean_layer_2/biasVarHandleOp*
shape:*"
shared_namemean_layer_2/bias*
dtype0*
_output_shapes
: 
s
%mean_layer_2/bias/Read/ReadVariableOpReadVariableOpmean_layer_2/bias*
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
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
�
(	arguments
)_variable_dict
*_trainable_weights
+_non_trainable_weights
,	variables
-trainable_variables
.regularization_losses
/	keras_api
8
0
1
2
3
4
5
"6
#7
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
�
0metrics
	variables
1non_trainable_variables
2layer_regularization_losses

3layers
regularization_losses
	trainable_variables
 
 
 
 
�
4metrics
	variables
5non_trainable_variables
trainable_variables

6layers
regularization_losses
7layer_regularization_losses
^\
VARIABLE_VALUEbaselayer_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbaselayer_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
8metrics
	variables
9non_trainable_variables
trainable_variables

:layers
regularization_losses
;layer_regularization_losses
db
VARIABLE_VALUEdense_var_layer_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEdense_var_layer_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
<metrics
	variables
=non_trainable_variables
trainable_variables

>layers
regularization_losses
?layer_regularization_losses
][
VARIABLE_VALUEpi_layer_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEpi_layer_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
�
@metrics
	variables
Anon_trainable_variables
trainable_variables

Blayers
 regularization_losses
Clayer_regularization_losses
_]
VARIABLE_VALUEmean_layer_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEmean_layer_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
�
Dmetrics
$	variables
Enon_trainable_variables
%trainable_variables

Flayers
&regularization_losses
Glayer_regularization_losses
 
 
 
 
 
 
 
�
Hmetrics
,	variables
Inon_trainable_variables
-trainable_variables

Jlayers
.regularization_losses
Klayer_regularization_losses
 
 
 
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
 *
dtype0*
_output_shapes
: 
z
serving_default_input_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1baselayer_2/kernelbaselayer_2/biasdense_var_layer_2/kerneldense_var_layer_2/biasmean_layer_2/kernelmean_layer_2/biaspi_layer_2/kernelpi_layer_2/bias*/
_gradient_op_typePartitionedCall-38075572*/
f*R(
&__inference_signature_wrapper_38075360*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
_output_shapes;
9:���������:���������:���������
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&baselayer_2/kernel/Read/ReadVariableOp$baselayer_2/bias/Read/ReadVariableOp,dense_var_layer_2/kernel/Read/ReadVariableOp*dense_var_layer_2/bias/Read/ReadVariableOp%pi_layer_2/kernel/Read/ReadVariableOp#pi_layer_2/bias/Read/ReadVariableOp'mean_layer_2/kernel/Read/ReadVariableOp%mean_layer_2/bias/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-38075604**
f%R#
!__inference__traced_save_38075603*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2
*
_output_shapes
: 
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebaselayer_2/kernelbaselayer_2/biasdense_var_layer_2/kerneldense_var_layer_2/biaspi_layer_2/kernelpi_layer_2/biasmean_layer_2/kernelmean_layer_2/bias*/
_gradient_op_typePartitionedCall-38075641*-
f(R&
$__inference__traced_restore_38075640*
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
: ʑ
� 
�
C__inference_model_layer_call_and_return_conditional_losses_38075286

inputs,
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

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall�
!baselayer/StatefulPartitionedCallStatefulPartitionedCallinputs(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075118*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112*
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
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075145*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139*
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
variance_layer/PartitionedCallPartitionedCall0dense_var_layer/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-38075172*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075161*
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
_gradient_op_typePartitionedCall-38075201*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195*
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
_gradient_op_typePartitionedCall-38075229*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223*
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
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075161

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
�
C__inference_model_layer_call_and_return_conditional_losses_38075264
input_1,
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

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall�
!baselayer/StatefulPartitionedCallStatefulPartitionedCallinput_1(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075118*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112*
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
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075145*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139*
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
variance_layer/PartitionedCallPartitionedCall0dense_var_layer/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-38075180*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075168*
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
_gradient_op_typePartitionedCall-38075201*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195*
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
_gradient_op_typePartitionedCall-38075229*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223*
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
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall: : : : : :' #
!
_user_specified_name	input_1: : : 
�	
�
G__inference_baselayer_layer_call_and_return_conditional_losses_38075473

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
�
C__inference_model_layer_call_and_return_conditional_losses_38075243
input_1,
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

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall�
!baselayer/StatefulPartitionedCallStatefulPartitionedCallinput_1(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075118*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112*
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
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075145*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139*
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
variance_layer/PartitionedCallPartitionedCall0dense_var_layer/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-38075172*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075161*
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
_gradient_op_typePartitionedCall-38075201*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195*
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
_gradient_op_typePartitionedCall-38075229*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223*
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
_user_specified_name	input_1: : : 
�
M
1__inference_variance_layer_layer_call_fn_38075547

inputs
identity�
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-38075172*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075161*
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
�
�
,__inference_baselayer_layer_call_fn_38075480

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075118*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112*
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
�-
�
C__inference_model_layer_call_and_return_conditional_losses_38075395

inputs,
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

identity_2�� baselayer/BiasAdd/ReadVariableOp�baselayer/MatMul/ReadVariableOp�&dense_var_layer/BiasAdd/ReadVariableOp�%dense_var_layer/MatMul/ReadVariableOp�!mean_layer/BiasAdd/ReadVariableOp� mean_layer/MatMul/ReadVariableOp�pi_layer/BiasAdd/ReadVariableOp�pi_layer/MatMul/ReadVariableOp�
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2}
baselayer/MatMulMatMulinputs'baselayer/MatMul/ReadVariableOp:value:0*
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
:���������2�
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
dense_var_layer/MatMulMatMulbaselayer/Tanh:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
variance_layer/ExpExp dense_var_layer/BiasAdd:output:0*
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
!mean_layer/BiasAdd/ReadVariableOp!mean_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
�
�
(__inference_model_layer_call_fn_38075341
input_1"
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
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-38075326*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_38075325*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
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
_user_specified_name	input_1: : : 
�	
�
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112

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
(__inference_model_layer_call_fn_38075462

inputs"
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
_gradient_op_typePartitionedCall-38075326*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_38075325*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
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
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075542

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
�
M
1__inference_variance_layer_layer_call_fn_38075552

inputs
identity�
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-38075180*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075168*
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
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195

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
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�	
�
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075525

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
�
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075168

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
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075537

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
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139

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
�
C__inference_model_layer_call_and_return_conditional_losses_38075325

inputs,
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

identity_2��!baselayer/StatefulPartitionedCall�'dense_var_layer/StatefulPartitionedCall�"mean_layer/StatefulPartitionedCall� pi_layer/StatefulPartitionedCall�
!baselayer/StatefulPartitionedCallStatefulPartitionedCallinputs(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075118*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_38075112*
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
:���������2�
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075145*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139*
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
variance_layer/PartitionedCallPartitionedCall0dense_var_layer/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-38075180*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075168*
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
_gradient_op_typePartitionedCall-38075201*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195*
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
_gradient_op_typePartitionedCall-38075229*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223*
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
�	
�
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223

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
�
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075490

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
:���������2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
�
�
!__inference__traced_save_38075603
file_prefix1
-savev2_baselayer_2_kernel_read_readvariableop/
+savev2_baselayer_2_bias_read_readvariableop7
3savev2_dense_var_layer_2_kernel_read_readvariableop5
1savev2_dense_var_layer_2_bias_read_readvariableop0
,savev2_pi_layer_2_kernel_read_readvariableop.
*savev2_pi_layer_2_bias_read_readvariableop2
.savev2_mean_layer_2_kernel_read_readvariableop0
,savev2_mean_layer_2_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_c1c4a79489034e7eb0bd9e9048e4f788/part*
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_baselayer_2_kernel_read_readvariableop+savev2_baselayer_2_bias_read_readvariableop3savev2_dense_var_layer_2_kernel_read_readvariableop1savev2_dense_var_layer_2_bias_read_readvariableop,savev2_pi_layer_2_kernel_read_readvariableop*savev2_pi_layer_2_bias_read_readvariableop.savev2_mean_layer_2_kernel_read_readvariableop,savev2_mean_layer_2_bias_read_readvariableop"/device:CPU:0*
dtypes

2*
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
�-
�
C__inference_model_layer_call_and_return_conditional_losses_38075428

inputs,
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

identity_2�� baselayer/BiasAdd/ReadVariableOp�baselayer/MatMul/ReadVariableOp�&dense_var_layer/BiasAdd/ReadVariableOp�%dense_var_layer/MatMul/ReadVariableOp�!mean_layer/BiasAdd/ReadVariableOp� mean_layer/MatMul/ReadVariableOp�pi_layer/BiasAdd/ReadVariableOp�pi_layer/MatMul/ReadVariableOp�
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2}
baselayer/MatMulMatMulinputs'baselayer/MatMul/ReadVariableOp:value:0*
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
:���������2�
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
dense_var_layer/MatMulMatMulbaselayer/Tanh:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������m
variance_layer/ExpExp dense_var_layer/BiasAdd:output:0*
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
pi_layer/BiasAdd/ReadVariableOppi_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
�
�
(__inference_model_layer_call_fn_38075302
input_1"
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
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-38075287*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_38075286*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
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
_user_specified_name	input_1: : : 
�
�
+__inference_pi_layer_layer_call_fn_38075515

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075229*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075223*
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
�
�
-__inference_mean_layer_layer_call_fn_38075532

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075201*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075195*
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
�%
�
$__inference__traced_restore_38075640
file_prefix'
#assignvariableop_baselayer_2_kernel'
#assignvariableop_1_baselayer_2_bias/
+assignvariableop_2_dense_var_layer_2_kernel-
)assignvariableop_3_dense_var_layer_2_bias(
$assignvariableop_4_pi_layer_2_kernel&
"assignvariableop_5_pi_layer_2_bias*
&assignvariableop_6_mean_layer_2_kernel(
$assignvariableop_7_mean_layer_2_bias

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

2*4
_output_shapes"
 ::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp#assignvariableop_baselayer_2_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_baselayer_2_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_dense_var_layer_2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp)assignvariableop_3_dense_var_layer_2_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp$assignvariableop_4_pi_layer_2_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp"assignvariableop_5_pi_layer_2_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_mean_layer_2_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_mean_layer_2_biasIdentity_7:output:0*
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
�
�
2__inference_dense_var_layer_layer_call_fn_38075497

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-38075145*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075139*
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
�2
�
#__inference__wrapped_model_38075095
input_12
.model_baselayer_matmul_readvariableop_resource3
/model_baselayer_biasadd_readvariableop_resource8
4model_dense_var_layer_matmul_readvariableop_resource9
5model_dense_var_layer_biasadd_readvariableop_resource3
/model_mean_layer_matmul_readvariableop_resource4
0model_mean_layer_biasadd_readvariableop_resource1
-model_pi_layer_matmul_readvariableop_resource2
.model_pi_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2��&model/baselayer/BiasAdd/ReadVariableOp�%model/baselayer/MatMul/ReadVariableOp�,model/dense_var_layer/BiasAdd/ReadVariableOp�+model/dense_var_layer/MatMul/ReadVariableOp�'model/mean_layer/BiasAdd/ReadVariableOp�&model/mean_layer/MatMul/ReadVariableOp�%model/pi_layer/BiasAdd/ReadVariableOp�$model/pi_layer/MatMul/ReadVariableOp�
%model/baselayer/MatMul/ReadVariableOpReadVariableOp.model_baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model/baselayer/MatMulMatMulinput_1-model/baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2�
&model/baselayer/BiasAdd/ReadVariableOpReadVariableOp/model_baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2�
model/baselayer/BiasAddBiasAdd model/baselayer/MatMul:product:0.model/baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2p
model/baselayer/TanhTanh model/baselayer/BiasAdd:output:0*
T0*'
_output_shapes
:���������2�
+model/dense_var_layer/MatMul/ReadVariableOpReadVariableOp4model_dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model/dense_var_layer/MatMulMatMulmodel/baselayer/Tanh:y:03model/dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
,model/dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp5model_dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/dense_var_layer/BiasAddBiasAdd&model/dense_var_layer/MatMul:product:04model/dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������y
model/variance_layer/ExpExp&model/dense_var_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
&model/mean_layer/MatMul/ReadVariableOpReadVariableOp/model_mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model/mean_layer/MatMulMatMulmodel/baselayer/Tanh:y:0.model/mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'model/mean_layer/BiasAdd/ReadVariableOpReadVariableOp0model_mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/mean_layer/BiasAddBiasAdd!model/mean_layer/MatMul:product:0/model/mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
$model/pi_layer/MatMul/ReadVariableOpReadVariableOp-model_pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2�
model/pi_layer/MatMulMatMulmodel/baselayer/Tanh:y:0,model/pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
%model/pi_layer/BiasAdd/ReadVariableOpReadVariableOp.model_pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:�
model/pi_layer/BiasAddBiasAddmodel/pi_layer/MatMul:product:0-model/pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������t
model/pi_layer/SoftmaxSoftmaxmodel/pi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:����������
IdentityIdentity!model/mean_layer/BiasAdd:output:0'^model/baselayer/BiasAdd/ReadVariableOp&^model/baselayer/MatMul/ReadVariableOp-^model/dense_var_layer/BiasAdd/ReadVariableOp,^model/dense_var_layer/MatMul/ReadVariableOp(^model/mean_layer/BiasAdd/ReadVariableOp'^model/mean_layer/MatMul/ReadVariableOp&^model/pi_layer/BiasAdd/ReadVariableOp%^model/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_1Identity model/pi_layer/Softmax:softmax:0'^model/baselayer/BiasAdd/ReadVariableOp&^model/baselayer/MatMul/ReadVariableOp-^model/dense_var_layer/BiasAdd/ReadVariableOp,^model/dense_var_layer/MatMul/ReadVariableOp(^model/mean_layer/BiasAdd/ReadVariableOp'^model/mean_layer/MatMul/ReadVariableOp&^model/pi_layer/BiasAdd/ReadVariableOp%^model/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:����������

Identity_2Identitymodel/variance_layer/Exp:y:0'^model/baselayer/BiasAdd/ReadVariableOp&^model/baselayer/MatMul/ReadVariableOp-^model/dense_var_layer/BiasAdd/ReadVariableOp,^model/dense_var_layer/MatMul/ReadVariableOp(^model/mean_layer/BiasAdd/ReadVariableOp'^model/mean_layer/MatMul/ReadVariableOp&^model/pi_layer/BiasAdd/ReadVariableOp%^model/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:���������::::::::2R
'model/mean_layer/BiasAdd/ReadVariableOp'model/mean_layer/BiasAdd/ReadVariableOp2N
%model/pi_layer/BiasAdd/ReadVariableOp%model/pi_layer/BiasAdd/ReadVariableOp2Z
+model/dense_var_layer/MatMul/ReadVariableOp+model/dense_var_layer/MatMul/ReadVariableOp2\
,model/dense_var_layer/BiasAdd/ReadVariableOp,model/dense_var_layer/BiasAdd/ReadVariableOp2N
%model/baselayer/MatMul/ReadVariableOp%model/baselayer/MatMul/ReadVariableOp2L
$model/pi_layer/MatMul/ReadVariableOp$model/pi_layer/MatMul/ReadVariableOp2P
&model/mean_layer/MatMul/ReadVariableOp&model/mean_layer/MatMul/ReadVariableOp2P
&model/baselayer/BiasAdd/ReadVariableOp&model/baselayer/BiasAdd/ReadVariableOp: : : : : :' #
!
_user_specified_name	input_1: : : 
�
�
(__inference_model_layer_call_fn_38075445

inputs"
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
_gradient_op_typePartitionedCall-38075287*L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_38075286*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
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
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075508

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
&__inference_signature_wrapper_38075360
input_1"
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
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-38075345*,
f'R%
#__inference__wrapped_model_38075095*
Tout
2*-
config_proto

CPU

GPU2*0J 8*
Tin
2	*M
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
_user_specified_name	input_1: : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
pi_layer0
StatefulPartitionedCall:1���������>

mean_layer0
StatefulPartitionedCall:0���������B
variance_layer0
StatefulPartitionedCall:2���������tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:�
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
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
*L&call_and_return_all_conditional_losses
M_default_save_signature
N__call__"�.
_tf_keras_model�.{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float64", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT4EAQAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float64", "sparse": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT4EAQAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}}}
�
	variables
trainable_variables
regularization_losses
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"�
_tf_keras_layer�{"class_name": "InputLayer", "name": "input_1", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": [null, 5], "config": {"batch_input_shape": [null, 5], "dtype": "float64", "sparse": false, "name": "input_1"}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "baselayer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_var_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
*U&call_and_return_all_conditional_losses
V__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "pi_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
*W&call_and_return_all_conditional_losses
X__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "mean_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
�
(	arguments
)_variable_dict
*_trainable_weights
+_non_trainable_weights
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"�
_tf_keras_layer�{"class_name": "Lambda", "name": "variance_layer", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT4EAQAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
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
�
0metrics
	variables
1non_trainable_variables
2layer_regularization_losses

3layers
regularization_losses
	trainable_variables
N__call__
M_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
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
4metrics
	variables
5non_trainable_variables
trainable_variables

6layers
regularization_losses
7layer_regularization_losses
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
$:"22baselayer_2/kernel
:22baselayer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
8metrics
	variables
9non_trainable_variables
trainable_variables

:layers
regularization_losses
;layer_regularization_losses
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
*:(22dense_var_layer_2/kernel
$:"2dense_var_layer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
<metrics
	variables
=non_trainable_variables
trainable_variables

>layers
regularization_losses
?layer_regularization_losses
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
#:!22pi_layer_2/kernel
:2pi_layer_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
@metrics
	variables
Anon_trainable_variables
trainable_variables

Blayers
 regularization_losses
Clayer_regularization_losses
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
%:#22mean_layer_2/kernel
:2mean_layer_2/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dmetrics
$	variables
Enon_trainable_variables
%trainable_variables

Flayers
&regularization_losses
Glayer_regularization_losses
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
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
Hmetrics
,	variables
Inon_trainable_variables
-trainable_variables

Jlayers
.regularization_losses
Klayer_regularization_losses
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
�2�
C__inference_model_layer_call_and_return_conditional_losses_38075395
C__inference_model_layer_call_and_return_conditional_losses_38075428
C__inference_model_layer_call_and_return_conditional_losses_38075264
C__inference_model_layer_call_and_return_conditional_losses_38075243�
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
#__inference__wrapped_model_38075095�
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
input_1���������
�2�
(__inference_model_layer_call_fn_38075462
(__inference_model_layer_call_fn_38075302
(__inference_model_layer_call_fn_38075341
(__inference_model_layer_call_fn_38075445�
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
G__inference_baselayer_layer_call_and_return_conditional_losses_38075473�
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
,__inference_baselayer_layer_call_fn_38075480�
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
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075490�
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
2__inference_dense_var_layer_layer_call_fn_38075497�
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
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075508�
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
+__inference_pi_layer_layer_call_fn_38075515�
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
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075525�
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
-__inference_mean_layer_layer_call_fn_38075532�
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
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075542
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075537�
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
1__inference_variance_layer_layer_call_fn_38075547
1__inference_variance_layer_layer_call_fn_38075552�
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
&__inference_signature_wrapper_38075360input_1�
G__inference_baselayer_layer_call_and_return_conditional_losses_38075473\/�,
%�"
 �
inputs���������
� "%�"
�
0���������2
� �
&__inference_signature_wrapper_38075360�"#;�8
� 
1�.
,
input_1!�
input_1���������"���
.
pi_layer"�
pi_layer���������
2

mean_layer$�!

mean_layer���������
:
variance_layer(�%
variance_layer����������
(__inference_model_layer_call_fn_38075445�"#7�4
-�*
 �
inputs���������
p

 
� "Z�W
�
0���������
�
1���������
�
2����������
#__inference__wrapped_model_38075095�"#0�-
&�#
!�
input_1���������
� "���
.
pi_layer"�
pi_layer���������
2

mean_layer$�!

mean_layer���������
:
variance_layer(�%
variance_layer���������~
+__inference_pi_layer_layer_call_fn_38075515O/�,
%�"
 �
inputs���������2
� "�����������
1__inference_variance_layer_layer_call_fn_38075552S7�4
-�*
 �
inputs���������

 
p 
� "�����������
1__inference_variance_layer_layer_call_fn_38075547S7�4
-�*
 �
inputs���������

 
p
� "�����������
(__inference_model_layer_call_fn_38075341�"#8�5
.�+
!�
input_1���������
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
C__inference_model_layer_call_and_return_conditional_losses_38075243�"#8�5
.�+
!�
input_1���������
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
(__inference_model_layer_call_fn_38075462�"#7�4
-�*
 �
inputs���������
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
(__inference_model_layer_call_fn_38075302�"#8�5
.�+
!�
input_1���������
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
C__inference_model_layer_call_and_return_conditional_losses_38075264�"#8�5
.�+
!�
input_1���������
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
H__inference_mean_layer_layer_call_and_return_conditional_losses_38075525\"#/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� 
,__inference_baselayer_layer_call_fn_38075480O/�,
%�"
 �
inputs���������
� "����������2�
C__inference_model_layer_call_and_return_conditional_losses_38075428�"#7�4
-�*
 �
inputs���������
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
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_38075490\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075542`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� �
L__inference_variance_layer_layer_call_and_return_conditional_losses_38075537`7�4
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
C__inference_model_layer_call_and_return_conditional_losses_38075395�"#7�4
-�*
 �
inputs���������
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
2__inference_dense_var_layer_layer_call_fn_38075497O/�,
%�"
 �
inputs���������2
� "�����������
F__inference_pi_layer_layer_call_and_return_conditional_losses_38075508\/�,
%�"
 �
inputs���������2
� "%�"
�
0���������
� �
-__inference_mean_layer_layer_call_fn_38075532O"#/�,
%�"
 �
inputs���������2
� "����������