ЇЅ
§
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
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02unknown8мс

baselayer_6584/kernelVarHandleOp*
shape
:2*&
shared_namebaselayer_6584/kernel*
dtype0*
_output_shapes
: 

)baselayer_6584/kernel/Read/ReadVariableOpReadVariableOpbaselayer_6584/kernel*
dtype0*
_output_shapes

:2
~
baselayer_6584/biasVarHandleOp*
shape:2*$
shared_namebaselayer_6584/bias*
dtype0*
_output_shapes
: 
w
'baselayer_6584/bias/Read/ReadVariableOpReadVariableOpbaselayer_6584/bias*
dtype0*
_output_shapes
:2

dense_var_layer_6584/kernelVarHandleOp*
shape
:2*,
shared_namedense_var_layer_6584/kernel*
dtype0*
_output_shapes
: 

/dense_var_layer_6584/kernel/Read/ReadVariableOpReadVariableOpdense_var_layer_6584/kernel*
dtype0*
_output_shapes

:2

dense_var_layer_6584/biasVarHandleOp*
shape:**
shared_namedense_var_layer_6584/bias*
dtype0*
_output_shapes
: 

-dense_var_layer_6584/bias/Read/ReadVariableOpReadVariableOpdense_var_layer_6584/bias*
dtype0*
_output_shapes
:

pi_layer_6584/kernelVarHandleOp*
shape
:2*%
shared_namepi_layer_6584/kernel*
dtype0*
_output_shapes
: 
}
(pi_layer_6584/kernel/Read/ReadVariableOpReadVariableOppi_layer_6584/kernel*
dtype0*
_output_shapes

:2
|
pi_layer_6584/biasVarHandleOp*
shape:*#
shared_namepi_layer_6584/bias*
dtype0*
_output_shapes
: 
u
&pi_layer_6584/bias/Read/ReadVariableOpReadVariableOppi_layer_6584/bias*
dtype0*
_output_shapes
:

mean_layer_6584/kernelVarHandleOp*
shape
:2*'
shared_namemean_layer_6584/kernel*
dtype0*
_output_shapes
: 

*mean_layer_6584/kernel/Read/ReadVariableOpReadVariableOpmean_layer_6584/kernel*
dtype0*
_output_shapes

:2

mean_layer_6584/biasVarHandleOp*
shape:*%
shared_namemean_layer_6584/bias*
dtype0*
_output_shapes
: 
y
(mean_layer_6584/bias/Read/ReadVariableOpReadVariableOpmean_layer_6584/bias*
dtype0*
_output_shapes
:

NoOpNoOp
ё
ConstConst"/device:CPU:0*Ќ
valueЂB B

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
Љ
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

regularization_losses
0non_trainable_variables
1layer_regularization_losses
trainable_variables

2layers
3metrics
		variables
 
 
 
 

regularization_losses
4non_trainable_variables
5layer_regularization_losses
trainable_variables

6layers
7metrics
	variables
a_
VARIABLE_VALUEbaselayer_6584/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEbaselayer_6584/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses
8non_trainable_variables
9layer_regularization_losses
trainable_variables

:layers
;metrics
	variables
ge
VARIABLE_VALUEdense_var_layer_6584/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEdense_var_layer_6584/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses
<non_trainable_variables
=layer_regularization_losses
trainable_variables

>layers
?metrics
	variables
`^
VARIABLE_VALUEpi_layer_6584/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEpi_layer_6584/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables

Blayers
Cmetrics
 	variables
b`
VARIABLE_VALUEmean_layer_6584/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEmean_layer_6584/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1

$regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
%trainable_variables

Flayers
Gmetrics
&	variables
 
 
 
 
 
 
 

,regularization_losses
Hnon_trainable_variables
Ilayer_regularization_losses
-trainable_variables

Jlayers
Kmetrics
.	variables
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
 
 *
dtype0*
_output_shapes
: 
{
serving_default_input_11Placeholder*
shape:џџџџџџџџџ*
dtype0*'
_output_shapes
:џџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_11baselayer_6584/kernelbaselayer_6584/biasdense_var_layer_6584/kerneldense_var_layer_6584/biasmean_layer_6584/kernelmean_layer_6584/biaspi_layer_6584/kernelpi_layer_6584/bias*/
_gradient_op_typePartitionedCall-82676880*/
f*R(
&__inference_signature_wrapper_82676662*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
ё
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)baselayer_6584/kernel/Read/ReadVariableOp'baselayer_6584/bias/Read/ReadVariableOp/dense_var_layer_6584/kernel/Read/ReadVariableOp-dense_var_layer_6584/bias/Read/ReadVariableOp(pi_layer_6584/kernel/Read/ReadVariableOp&pi_layer_6584/bias/Read/ReadVariableOp*mean_layer_6584/kernel/Read/ReadVariableOp(mean_layer_6584/bias/Read/ReadVariableOpConst*/
_gradient_op_typePartitionedCall-82676912**
f%R#
!__inference__traced_save_82676911*
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
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebaselayer_6584/kernelbaselayer_6584/biasdense_var_layer_6584/kerneldense_var_layer_6584/biaspi_layer_6584/kernelpi_layer_6584/biasmean_layer_6584/kernelmean_layer_6584/bias*/
_gradient_op_typePartitionedCall-82676949*-
f(R&
$__inference__traced_restore_82676948*
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
: ьЈ
#

F__inference_model_10_layer_call_and_return_conditional_losses_82676627

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

identity_2Ђ!baselayer/StatefulPartitionedCallЂ'dense_var_layer/StatefulPartitionedCallЂ"mean_layer/StatefulPartitionedCallЂ pi_layer/StatefulPartitionedCall_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676409*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403*
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
:џџџџџџџџџ2
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2М
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676437*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431*
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
:џџџџџџџџџ
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЧ
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-82676473*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676461*
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
:џџџџџџџџџК
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676494*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488*
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
:џџџџџџџџџВ
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676522*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516*
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
:џџџџџџџџџ
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
з
Є
+__inference_model_10_layer_call_fn_82676753

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

identity_2ЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-82676586*O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_82676585*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
Ъ
M
1__inference_variance_layer_layer_call_fn_82676855

inputs
identityЇ
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-82676465*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676454*
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
:џџџџџџџџџ`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
з
Є
+__inference_model_10_layer_call_fn_82676770

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

identity_2ЂStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-82676628*O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_82676627*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
	
ц
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676798

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
х
Ў
-__inference_mean_layer_layer_call_fn_82676840

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676494*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488*
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
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
№
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676461

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
н
І
+__inference_model_10_layer_call_fn_82676643
input_11"
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

identity_2ЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinput_11statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-82676628*O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_82676627*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :( $
"
_user_specified_name
input_11: : : 
я
Г
2__inference_dense_var_layer_layer_call_fn_82676805

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676437*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431*
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
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
&
ќ
$__inference__traced_restore_82676948
file_prefix*
&assignvariableop_baselayer_6584_kernel*
&assignvariableop_1_baselayer_6584_bias2
.assignvariableop_2_dense_var_layer_6584_kernel0
,assignvariableop_3_dense_var_layer_6584_bias+
'assignvariableop_4_pi_layer_6584_kernel)
%assignvariableop_5_pi_layer_6584_bias-
)assignvariableop_6_mean_layer_6584_kernel+
'assignvariableop_7_mean_layer_6584_bias

identity_9ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7Ђ	RestoreV2ЂRestoreV2_1Ї
RestoreV2/tensor_namesConst"/device:CPU:0*Э
valueУBРB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:
RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:Ц
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes

2*4
_output_shapes"
 ::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp&assignvariableop_baselayer_6584_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_baselayer_6584_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_dense_var_layer_6584_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp,assignvariableop_3_dense_var_layer_6584_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp'assignvariableop_4_pi_layer_6584_kernelIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp%assignvariableop_5_pi_layer_6584_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp)assignvariableop_6_mean_layer_6584_kernelIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp'assignvariableop_7_mean_layer_6584_biasIdentity_7:output:0*
dtype0*
_output_shapes
 
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
:Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 

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
№
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676850

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
	
с
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676833

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
Щ	
р
G__inference_baselayer_layer_call_and_return_conditional_losses_82676781

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 

И
!__inference__traced_save_82676911
file_prefix4
0savev2_baselayer_6584_kernel_read_readvariableop2
.savev2_baselayer_6584_bias_read_readvariableop:
6savev2_dense_var_layer_6584_kernel_read_readvariableop8
4savev2_dense_var_layer_6584_bias_read_readvariableop3
/savev2_pi_layer_6584_kernel_read_readvariableop1
-savev2_pi_layer_6584_bias_read_readvariableop5
1savev2_mean_layer_6584_kernel_read_readvariableop3
/savev2_mean_layer_6584_bias_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_9c7189a8bec44797ab846f0860d2c076/part*
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
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Є
SaveV2/tensor_namesConst"/device:CPU:0*Э
valueУBРB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:}
SaveV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:Н
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_baselayer_6584_kernel_read_readvariableop.savev2_baselayer_6584_bias_read_readvariableop6savev2_dense_var_layer_6584_kernel_read_readvariableop4savev2_dense_var_layer_6584_bias_read_readvariableop/savev2_pi_layer_6584_kernel_read_readvariableop-savev2_pi_layer_6584_bias_read_readvariableop1savev2_mean_layer_6584_kernel_read_readvariableop/savev2_mean_layer_6584_bias_read_readvariableop"/device:CPU:0*
dtypes

2*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
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
:У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
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
№
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676454

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
з	
п
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
0
Ѕ
F__inference_model_10_layer_call_and_return_conditional_losses_82676700

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

identity_2Ђ baselayer/BiasAdd/ReadVariableOpЂbaselayer/MatMul/ReadVariableOpЂ&dense_var_layer/BiasAdd/ReadVariableOpЂ%dense_var_layer/MatMul/ReadVariableOpЂ!mean_layer/BiasAdd/ReadVariableOpЂ mean_layer/MatMul/ReadVariableOpЂpi_layer/BiasAdd/ReadVariableOpЂpi_layer/MatMul/ReadVariableOp_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЖ
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
baselayer/MatMulMatMulbaselayer/Cast:y:0'baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Д
 baselayer/BiasAdd/ReadVariableOpReadVariableOp)baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2
baselayer/BiasAddBiasAddbaselayer/MatMul:product:0(baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
baselayer/TanhTanhbaselayer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2q
dense_var_layer/CastCastbaselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2Т
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
dense_var_layer/MatMulMatMuldense_var_layer/Cast:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџР
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:І
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
variance_layer/CastCast dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџd
variance_layer/ExpExpvariance_layer/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџИ
 mean_layer/MatMul/ReadVariableOpReadVariableOp)mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
mean_layer/MatMulMatMulbaselayer/Tanh:y:0(mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
!mean_layer/BiasAdd/ReadVariableOpReadVariableOp*mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
mean_layer/BiasAddBiasAddmean_layer/MatMul:product:0)mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџД
pi_layer/MatMul/ReadVariableOpReadVariableOp'pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
pi_layer/MatMulMatMulbaselayer/Tanh:y:0&pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџВ
pi_layer/BiasAdd/ReadVariableOpReadVariableOp(pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
pi_layer/BiasAddBiasAddpi_layer/MatMul:product:0'pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
pi_layer/SoftmaxSoftmaxpi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitypi_layer/Softmax:softmax:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identitymean_layer/BiasAdd:output:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identityvariance_layer/Exp:y:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2@
pi_layer/MatMul/ReadVariableOppi_layer/MatMul/ReadVariableOp2D
 mean_layer/MatMul/ReadVariableOp mean_layer/MatMul/ReadVariableOp2D
 baselayer/BiasAdd/ReadVariableOp baselayer/BiasAdd/ReadVariableOp2F
!mean_layer/BiasAdd/ReadVariableOp!mean_layer/BiasAdd/ReadVariableOp2B
pi_layer/BiasAdd/ReadVariableOppi_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
в7

#__inference__wrapped_model_82676385
input_115
1model_10_baselayer_matmul_readvariableop_resource6
2model_10_baselayer_biasadd_readvariableop_resource;
7model_10_dense_var_layer_matmul_readvariableop_resource<
8model_10_dense_var_layer_biasadd_readvariableop_resource6
2model_10_mean_layer_matmul_readvariableop_resource7
3model_10_mean_layer_biasadd_readvariableop_resource4
0model_10_pi_layer_matmul_readvariableop_resource5
1model_10_pi_layer_biasadd_readvariableop_resource
identity

identity_1

identity_2Ђ)model_10/baselayer/BiasAdd/ReadVariableOpЂ(model_10/baselayer/MatMul/ReadVariableOpЂ/model_10/dense_var_layer/BiasAdd/ReadVariableOpЂ.model_10/dense_var_layer/MatMul/ReadVariableOpЂ*model_10/mean_layer/BiasAdd/ReadVariableOpЂ)model_10/mean_layer/MatMul/ReadVariableOpЂ(model_10/pi_layer/BiasAdd/ReadVariableOpЂ'model_10/pi_layer/MatMul/ReadVariableOpj
model_10/baselayer/CastCastinput_11*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџШ
(model_10/baselayer/MatMul/ReadVariableOpReadVariableOp1model_10_baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2Є
model_10/baselayer/MatMulMatMulmodel_10/baselayer/Cast:y:00model_10/baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Ц
)model_10/baselayer/BiasAdd/ReadVariableOpReadVariableOp2model_10_baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2Џ
model_10/baselayer/BiasAddBiasAdd#model_10/baselayer/MatMul:product:01model_10/baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2v
model_10/baselayer/TanhTanh#model_10/baselayer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
model_10/dense_var_layer/CastCastmodel_10/baselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2д
.model_10/dense_var_layer/MatMul/ReadVariableOpReadVariableOp7model_10_dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2Ж
model_10/dense_var_layer/MatMulMatMul!model_10/dense_var_layer/Cast:y:06model_10/dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџв
/model_10/dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp8model_10_dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:С
 model_10/dense_var_layer/BiasAddBiasAdd)model_10/dense_var_layer/MatMul:product:07model_10/dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
model_10/variance_layer/CastCast)model_10/dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџv
model_10/variance_layer/ExpExp model_10/variance_layer/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџЪ
)model_10/mean_layer/MatMul/ReadVariableOpReadVariableOp2model_10_mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2І
model_10/mean_layer/MatMulMatMulmodel_10/baselayer/Tanh:y:01model_10/mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџШ
*model_10/mean_layer/BiasAdd/ReadVariableOpReadVariableOp3model_10_mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:В
model_10/mean_layer/BiasAddBiasAdd$model_10/mean_layer/MatMul:product:02model_10/mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЦ
'model_10/pi_layer/MatMul/ReadVariableOpReadVariableOp0model_10_pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2Ђ
model_10/pi_layer/MatMulMatMulmodel_10/baselayer/Tanh:y:0/model_10/pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџФ
(model_10/pi_layer/BiasAdd/ReadVariableOpReadVariableOp1model_10_pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Ќ
model_10/pi_layer/BiasAddBiasAdd"model_10/pi_layer/MatMul:product:00model_10/pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџz
model_10/pi_layer/SoftmaxSoftmax"model_10/pi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџд
IdentityIdentity$model_10/mean_layer/BiasAdd:output:0*^model_10/baselayer/BiasAdd/ReadVariableOp)^model_10/baselayer/MatMul/ReadVariableOp0^model_10/dense_var_layer/BiasAdd/ReadVariableOp/^model_10/dense_var_layer/MatMul/ReadVariableOp+^model_10/mean_layer/BiasAdd/ReadVariableOp*^model_10/mean_layer/MatMul/ReadVariableOp)^model_10/pi_layer/BiasAdd/ReadVariableOp(^model_10/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџе

Identity_1Identity#model_10/pi_layer/Softmax:softmax:0*^model_10/baselayer/BiasAdd/ReadVariableOp)^model_10/baselayer/MatMul/ReadVariableOp0^model_10/dense_var_layer/BiasAdd/ReadVariableOp/^model_10/dense_var_layer/MatMul/ReadVariableOp+^model_10/mean_layer/BiasAdd/ReadVariableOp*^model_10/mean_layer/MatMul/ReadVariableOp)^model_10/pi_layer/BiasAdd/ReadVariableOp(^model_10/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџб

Identity_2Identitymodel_10/variance_layer/Exp:y:0*^model_10/baselayer/BiasAdd/ReadVariableOp)^model_10/baselayer/MatMul/ReadVariableOp0^model_10/dense_var_layer/BiasAdd/ReadVariableOp/^model_10/dense_var_layer/MatMul/ReadVariableOp+^model_10/mean_layer/BiasAdd/ReadVariableOp*^model_10/mean_layer/MatMul/ReadVariableOp)^model_10/pi_layer/BiasAdd/ReadVariableOp(^model_10/pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2b
/model_10/dense_var_layer/BiasAdd/ReadVariableOp/model_10/dense_var_layer/BiasAdd/ReadVariableOp2`
.model_10/dense_var_layer/MatMul/ReadVariableOp.model_10/dense_var_layer/MatMul/ReadVariableOp2T
(model_10/baselayer/MatMul/ReadVariableOp(model_10/baselayer/MatMul/ReadVariableOp2V
)model_10/mean_layer/MatMul/ReadVariableOp)model_10/mean_layer/MatMul/ReadVariableOp2R
'model_10/pi_layer/MatMul/ReadVariableOp'model_10/pi_layer/MatMul/ReadVariableOp2V
)model_10/baselayer/BiasAdd/ReadVariableOp)model_10/baselayer/BiasAdd/ReadVariableOp2X
*model_10/mean_layer/BiasAdd/ReadVariableOp*model_10/mean_layer/BiasAdd/ReadVariableOp2T
(model_10/pi_layer/BiasAdd/ReadVariableOp(model_10/pi_layer/BiasAdd/ReadVariableOp: : : : : :( $
"
_user_specified_name
input_11: : : 
Щ	
р
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
0
Ѕ
F__inference_model_10_layer_call_and_return_conditional_losses_82676736

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

identity_2Ђ baselayer/BiasAdd/ReadVariableOpЂbaselayer/MatMul/ReadVariableOpЂ&dense_var_layer/BiasAdd/ReadVariableOpЂ%dense_var_layer/MatMul/ReadVariableOpЂ!mean_layer/BiasAdd/ReadVariableOpЂ mean_layer/MatMul/ReadVariableOpЂpi_layer/BiasAdd/ReadVariableOpЂpi_layer/MatMul/ReadVariableOp_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЖ
baselayer/MatMul/ReadVariableOpReadVariableOp(baselayer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
baselayer/MatMulMatMulbaselayer/Cast:y:0'baselayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2Д
 baselayer/BiasAdd/ReadVariableOpReadVariableOp)baselayer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:2
baselayer/BiasAddBiasAddbaselayer/MatMul:product:0(baselayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ2d
baselayer/TanhTanhbaselayer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ2q
dense_var_layer/CastCastbaselayer/Tanh:y:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2Т
%dense_var_layer/MatMul/ReadVariableOpReadVariableOp.dense_var_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
dense_var_layer/MatMulMatMuldense_var_layer/Cast:y:0-dense_var_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџР
&dense_var_layer/BiasAdd/ReadVariableOpReadVariableOp/dense_var_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:І
dense_var_layer/BiasAddBiasAdd dense_var_layer/MatMul:product:0.dense_var_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ~
variance_layer/CastCast dense_var_layer/BiasAdd:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџd
variance_layer/ExpExpvariance_layer/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџИ
 mean_layer/MatMul/ReadVariableOpReadVariableOp)mean_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
mean_layer/MatMulMatMulbaselayer/Tanh:y:0(mean_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЖ
!mean_layer/BiasAdd/ReadVariableOpReadVariableOp*mean_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
mean_layer/BiasAddBiasAddmean_layer/MatMul:product:0)mean_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџД
pi_layer/MatMul/ReadVariableOpReadVariableOp'pi_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2
pi_layer/MatMulMatMulbaselayer/Tanh:y:0&pi_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџВ
pi_layer/BiasAdd/ReadVariableOpReadVariableOp(pi_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:
pi_layer/BiasAddBiasAddpi_layer/MatMul:product:0'pi_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџh
pi_layer/SoftmaxSoftmaxpi_layer/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitypi_layer/Softmax:softmax:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identitymean_layer/BiasAdd:output:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identityvariance_layer/Exp:y:0!^baselayer/BiasAdd/ReadVariableOp ^baselayer/MatMul/ReadVariableOp'^dense_var_layer/BiasAdd/ReadVariableOp&^dense_var_layer/MatMul/ReadVariableOp"^mean_layer/BiasAdd/ReadVariableOp!^mean_layer/MatMul/ReadVariableOp ^pi_layer/BiasAdd/ReadVariableOp^pi_layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2@
pi_layer/MatMul/ReadVariableOppi_layer/MatMul/ReadVariableOp2D
 mean_layer/MatMul/ReadVariableOp mean_layer/MatMul/ReadVariableOp2D
 baselayer/BiasAdd/ReadVariableOp baselayer/BiasAdd/ReadVariableOp2B
pi_layer/BiasAdd/ReadVariableOppi_layer/BiasAdd/ReadVariableOp2F
!mean_layer/BiasAdd/ReadVariableOp!mean_layer/BiasAdd/ReadVariableOp2P
&dense_var_layer/BiasAdd/ReadVariableOp&dense_var_layer/BiasAdd/ReadVariableOp2N
%dense_var_layer/MatMul/ReadVariableOp%dense_var_layer/MatMul/ReadVariableOp2B
baselayer/MatMul/ReadVariableOpbaselayer/MatMul/ReadVariableOp: : : : : :& "
 
_user_specified_nameinputs: : : 
№
h
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676845

inputs
identityD
ExpExpinputs*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
з	
п
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676816

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
#

F__inference_model_10_layer_call_and_return_conditional_losses_82676560
input_11,
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

identity_2Ђ!baselayer/StatefulPartitionedCallЂ'dense_var_layer/StatefulPartitionedCallЂ"mean_layer/StatefulPartitionedCallЂ pi_layer/StatefulPartitionedCalla
baselayer/CastCastinput_11*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676409*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403*
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
:џџџџџџџџџ2
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2М
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676437*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431*
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
:џџџџџџџџџ
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЧ
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-82676473*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676461*
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
:џџџџџџџџџК
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676494*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488*
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
:џџџџџџџџџВ
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676522*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516*
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
:џџџџџџџџџ
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall: : : : : :( $
"
_user_specified_name
input_11: : : 
Ъ
M
1__inference_variance_layer_layer_call_fn_82676860

inputs
identityЇ
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-82676473*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676461*
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
:џџџџџџџџџ`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*&
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
с
Ќ
+__inference_pi_layer_layer_call_fn_82676823

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676522*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516*
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
:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
у
­
,__inference_baselayer_layer_call_fn_82676788

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676409*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403*
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
:џџџџџџџџџ2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ2"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
Е
Ё
&__inference_signature_wrapper_82676662
input_11"
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

identity_2ЂStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinput_11statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-82676647*,
f'R%
#__inference__wrapped_model_82676385*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :( $
"
_user_specified_name
input_11: : : 
н
І
+__inference_model_10_layer_call_fn_82676601
input_11"
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

identity_2ЂStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinput_11statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*/
_gradient_op_typePartitionedCall-82676586*O
fJRH
F__inference_model_10_layer_call_and_return_conditional_losses_82676585*
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
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :( $
"
_user_specified_name
input_11: : : 
	
ц
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
#

F__inference_model_10_layer_call_and_return_conditional_losses_82676585

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

identity_2Ђ!baselayer/StatefulPartitionedCallЂ'dense_var_layer/StatefulPartitionedCallЂ"mean_layer/StatefulPartitionedCallЂ pi_layer/StatefulPartitionedCall_
baselayer/CastCastinputs*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676409*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403*
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
:џџџџџџџџџ2
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2М
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676437*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431*
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
:џџџџџџџџџ
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЧ
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-82676465*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676454*
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
:џџџџџџџџџК
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676494*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488*
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
:џџџџџџџџџВ
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676522*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516*
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
:џџџџџџџџџ
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall: : : : : :& "
 
_user_specified_nameinputs: : : 
#

F__inference_model_10_layer_call_and_return_conditional_losses_82676536
input_11,
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

identity_2Ђ!baselayer/StatefulPartitionedCallЂ'dense_var_layer/StatefulPartitionedCallЂ"mean_layer/StatefulPartitionedCallЂ pi_layer/StatefulPartitionedCalla
baselayer/CastCastinput_11*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ
!baselayer/StatefulPartitionedCallStatefulPartitionedCallbaselayer/Cast:y:0(baselayer_statefulpartitionedcall_args_1(baselayer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676409*P
fKRI
G__inference_baselayer_layer_call_and_return_conditional_losses_82676403*
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
:џџџџџџџџџ2
dense_var_layer/CastCast*baselayer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџ2М
'dense_var_layer/StatefulPartitionedCallStatefulPartitionedCalldense_var_layer/Cast:y:0.dense_var_layer_statefulpartitionedcall_args_1.dense_var_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676437*V
fQRO
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676431*
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
:џџџџџџџџџ
variance_layer/CastCast0dense_var_layer/StatefulPartitionedCall:output:0*

SrcT0*

DstT0*'
_output_shapes
:џџџџџџџџџЧ
variance_layer/PartitionedCallPartitionedCallvariance_layer/Cast:y:0*/
_gradient_op_typePartitionedCall-82676465*U
fPRN
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676454*
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
:џџџџџџџџџК
"mean_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0)mean_layer_statefulpartitionedcall_args_1)mean_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676494*Q
fLRJ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488*
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
:џџџџџџџџџВ
 pi_layer/StatefulPartitionedCallStatefulPartitionedCall*baselayer/StatefulPartitionedCall:output:0'pi_layer_statefulpartitionedcall_args_1'pi_layer_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-82676522*O
fJRH
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676516*
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
:џџџџџџџџџ
IdentityIdentity)pi_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_1Identity+mean_layer/StatefulPartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ

Identity_2Identity'variance_layer/PartitionedCall:output:0"^baselayer/StatefulPartitionedCall(^dense_var_layer/StatefulPartitionedCall#^mean_layer/StatefulPartitionedCall!^pi_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*F
_input_shapes5
3:џџџџџџџџџ::::::::2R
'dense_var_layer/StatefulPartitionedCall'dense_var_layer/StatefulPartitionedCall2F
!baselayer/StatefulPartitionedCall!baselayer/StatefulPartitionedCall2D
 pi_layer/StatefulPartitionedCall pi_layer/StatefulPartitionedCall2H
"mean_layer/StatefulPartitionedCall"mean_layer/StatefulPartitionedCall: : : : : :( $
"
_user_specified_name
input_11: : : 
	
с
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676488

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЂ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:2i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџ2::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*Б
serving_default
=
input_111
serving_default_input_11:0џџџџџџџџџ<
pi_layer0
StatefulPartitionedCall:1џџџџџџџџџ>

mean_layer0
StatefulPartitionedCall:0џџџџџџџџџB
variance_layer0
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ФЛ
р1
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
L_default_save_signature
*M&call_and_return_all_conditional_losses
N__call__"ю.
_tf_keras_modelд.{"class_name": "Model", "name": "model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT7lAAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_11"}, "name": "input_11", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "baselayer", "inbound_nodes": [[["input_11", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_var_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "pi_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "mean_layer", "inbound_nodes": [[["baselayer", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT7lAAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "variance_layer", "inbound_nodes": [[["dense_var_layer", 0, 0, {}]]]}], "input_layers": [["input_11", 0, 0]], "output_layers": [["pi_layer", 0, 0], ["mean_layer", 0, 0], ["variance_layer", 0, 0]]}}}
Ѓ
regularization_losses
trainable_variables
	variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"
_tf_keras_layerњ{"class_name": "InputLayer", "name": "input_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 5], "config": {"batch_input_shape": [null, 5], "dtype": "float32", "sparse": false, "name": "input_11"}}
є

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"Я
_tf_keras_layerЕ{"class_name": "Dense", "name": "baselayer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "baselayer", "trainable": true, "dtype": "float64", "units": 50, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}}


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"н
_tf_keras_layerУ{"class_name": "Dense", "name": "dense_var_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_var_layer", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
ѕ

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
*U&call_and_return_all_conditional_losses
V__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "pi_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "pi_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
ј

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
*W&call_and_return_all_conditional_losses
X__call__"г
_tf_keras_layerЙ{"class_name": "Dense", "name": "mean_layer", "trainable": true, "expects_training_arg": false, "dtype": "float64", "batch_input_shape": null, "config": {"name": "mean_layer", "trainable": true, "dtype": "float64", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 50}}}}
Ч
(	arguments
)_variable_dict
*_trainable_weights
+_non_trainable_weights
,regularization_losses
-trainable_variables
.	variables
/	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"с
_tf_keras_layerЧ{"class_name": "Lambda", "name": "variance_layer", "trainable": true, "expects_training_arg": true, "dtype": "float64", "batch_input_shape": null, "config": {"name": "variance_layer", "trainable": true, "dtype": "float64", "function": ["4wEAAAAAAAAAAQAAAAMAAABDAAAAcwwAAAB0AGoBoAJ8AKEBUwApAU4pA9oCdGbaBG1hdGjaA2V4\ncCkB2gF4qQByBQAAAPorRjovUHJvZ3JhbW1pbmcvcHl0aG9uL0dhbGFDbHVzdGVyTW9kL2dtbS5w\nedoIPGxhbWJkYT7lAAAA8wAAAAA=\n", null, null], "function_type": "lambda", "module": "__main__", "output_shape": [4], "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
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
З
regularization_losses
0non_trainable_variables
1layer_regularization_losses
trainable_variables

2layers
3metrics
		variables
N__call__
L_default_save_signature
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

regularization_losses
4non_trainable_variables
5layer_regularization_losses
trainable_variables

6layers
7metrics
	variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
':%22baselayer_6584/kernel
!:22baselayer_6584/bias
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

regularization_losses
8non_trainable_variables
9layer_regularization_losses
trainable_variables

:layers
;metrics
	variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
-:+22dense_var_layer_6584/kernel
':%2dense_var_layer_6584/bias
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

regularization_losses
<non_trainable_variables
=layer_regularization_losses
trainable_variables

>layers
?metrics
	variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
&:$22pi_layer_6584/kernel
 :2pi_layer_6584/bias
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

regularization_losses
@non_trainable_variables
Alayer_regularization_losses
trainable_variables

Blayers
Cmetrics
 	variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
(:&22mean_layer_6584/kernel
": 2mean_layer_6584/bias
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

$regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
%trainable_variables

Flayers
Gmetrics
&	variables
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

,regularization_losses
Hnon_trainable_variables
Ilayer_regularization_losses
-trainable_variables

Jlayers
Kmetrics
.	variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
т2п
#__inference__wrapped_model_82676385З
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *'Ђ$
"
input_11џџџџџџџџџ
ц2у
F__inference_model_10_layer_call_and_return_conditional_losses_82676736
F__inference_model_10_layer_call_and_return_conditional_losses_82676700
F__inference_model_10_layer_call_and_return_conditional_losses_82676536
F__inference_model_10_layer_call_and_return_conditional_losses_82676560Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
њ2ї
+__inference_model_10_layer_call_fn_82676753
+__inference_model_10_layer_call_fn_82676643
+__inference_model_10_layer_call_fn_82676601
+__inference_model_10_layer_call_fn_82676770Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ё2ю
G__inference_baselayer_layer_call_and_return_conditional_losses_82676781Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_baselayer_layer_call_fn_82676788Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ї2є
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676798Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
м2й
2__inference_dense_var_layer_layer_call_fn_82676805Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676816Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_pi_layer_layer_call_fn_82676823Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђ2я
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676833Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_mean_layer_layer_call_fn_82676840Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
т2п
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676845
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676850Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ќ2Љ
1__inference_variance_layer_layer_call_fn_82676855
1__inference_variance_layer_layer_call_fn_82676860Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
6B4
&__inference_signature_wrapper_82676662input_11Я
+__inference_model_10_layer_call_fn_82676753"#7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџќ
F__inference_model_10_layer_call_and_return_conditional_losses_82676536Б"#9Ђ6
/Ђ,
"
input_11џџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 б
+__inference_model_10_layer_call_fn_82676643Ё"#9Ђ6
/Ђ,
"
input_11џџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЯ
+__inference_model_10_layer_call_fn_82676770"#7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџЈ
H__inference_mean_layer_layer_call_and_return_conditional_losses_82676833\"#/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 ќ
F__inference_model_10_layer_call_and_return_conditional_losses_82676560Б"#9Ђ6
/Ђ,
"
input_11џџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 б
+__inference_model_10_layer_call_fn_82676601Ё"#9Ђ6
/Ђ,
"
input_11џџџџџџџџџ
p

 
Њ "ZW

0џџџџџџџџџ

1џџџџџџџџџ

2џџџџџџџџџА
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676850`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "%Ђ"

0џџџџџџџџџ
 А
L__inference_variance_layer_layer_call_and_return_conditional_losses_82676845`7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "%Ђ"

0џџџџџџџџџ
 
,__inference_baselayer_layer_call_fn_82676788O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџ2
#__inference__wrapped_model_82676385ф"#1Ђ.
'Ђ$
"
input_11џџџџџџџџџ
Њ "ЄЊ 
.
pi_layer"
pi_layerџџџџџџџџџ
2

mean_layer$!

mean_layerџџџџџџџџџ
:
variance_layer(%
variance_layerџџџџџџџџџ­
M__inference_dense_var_layer_layer_call_and_return_conditional_losses_82676798\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 І
F__inference_pi_layer_layer_call_and_return_conditional_losses_82676816\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "%Ђ"

0џџџџџџџџџ
 
-__inference_mean_layer_layer_call_fn_82676840O"#/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџњ
F__inference_model_10_layer_call_and_return_conditional_losses_82676736Џ"#7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p 

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 Ї
G__inference_baselayer_layer_call_and_return_conditional_losses_82676781\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ2
 
&__inference_signature_wrapper_82676662№"#=Ђ:
Ђ 
3Њ0
.
input_11"
input_11џџџџџџџџџ"ЄЊ 
.
pi_layer"
pi_layerџџџџџџџџџ
2

mean_layer$!

mean_layerџџџџџџџџџ
:
variance_layer(%
variance_layerџџџџџџџџџ
2__inference_dense_var_layer_layer_call_fn_82676805O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ~
+__inference_pi_layer_layer_call_fn_82676823O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ2
Њ "џџџџџџџџџ
1__inference_variance_layer_layer_call_fn_82676855S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p
Њ "џџџџџџџџџ
1__inference_variance_layer_layer_call_fn_82676860S7Ђ4
-Ђ*
 
inputsџџџџџџџџџ

 
p 
Њ "џџџџџџџџџњ
F__inference_model_10_layer_call_and_return_conditional_losses_82676700Џ"#7Ђ4
-Ђ*
 
inputsџџџџџџџџџ
p

 
Њ "jЂg
`]

0/0џџџџџџџџџ

0/1џџџџџџџџџ

0/2џџџџџџџџџ
 