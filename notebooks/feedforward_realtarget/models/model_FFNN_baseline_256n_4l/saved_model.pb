╕╫	
Щ¤
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
dtypetypeИ
╛
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
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02unknown8ец
Е
Hidden-Layer1/kernelVarHandleOp*%
shared_nameHidden-Layer1/kernel*
_output_shapes
: *
dtype0*
shape:	А
~
(Hidden-Layer1/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer1/kernel*
_output_shapes
:	А*
dtype0
}
Hidden-Layer1/biasVarHandleOp*#
shared_nameHidden-Layer1/bias*
dtype0*
shape:А*
_output_shapes
: 
v
&Hidden-Layer1/bias/Read/ReadVariableOpReadVariableOpHidden-Layer1/bias*
dtype0*
_output_shapes	
:А
Ж
Hidden-Layer2/kernelVarHandleOp*%
shared_nameHidden-Layer2/kernel*
dtype0*
_output_shapes
: *
shape:
АА

(Hidden-Layer2/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer2/kernel* 
_output_shapes
:
АА*
dtype0
}
Hidden-Layer2/biasVarHandleOp*
_output_shapes
: *
shape:А*
dtype0*#
shared_nameHidden-Layer2/bias
v
&Hidden-Layer2/bias/Read/ReadVariableOpReadVariableOpHidden-Layer2/bias*
_output_shapes	
:А*
dtype0
Ж
Hidden-Layer3/kernelVarHandleOp*
_output_shapes
: *%
shared_nameHidden-Layer3/kernel*
dtype0*
shape:
АА

(Hidden-Layer3/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer3/kernel* 
_output_shapes
:
АА*
dtype0
}
Hidden-Layer3/biasVarHandleOp*
_output_shapes
: *#
shared_nameHidden-Layer3/bias*
dtype0*
shape:А
v
&Hidden-Layer3/bias/Read/ReadVariableOpReadVariableOpHidden-Layer3/bias*
_output_shapes	
:А*
dtype0
Ж
Hidden-Layer4/kernelVarHandleOp*
dtype0*
shape:
АА*
_output_shapes
: *%
shared_nameHidden-Layer4/kernel

(Hidden-Layer4/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer4/kernel* 
_output_shapes
:
АА*
dtype0
}
Hidden-Layer4/biasVarHandleOp*
dtype0*#
shared_nameHidden-Layer4/bias*
_output_shapes
: *
shape:А
v
&Hidden-Layer4/bias/Read/ReadVariableOpReadVariableOpHidden-Layer4/bias*
_output_shapes	
:А*
dtype0
З
Output-Layer_2/kernelVarHandleOp*&
shared_nameOutput-Layer_2/kernel*
dtype0*
_output_shapes
: *
shape:	А
А
)Output-Layer_2/kernel/Read/ReadVariableOpReadVariableOpOutput-Layer_2/kernel*
_output_shapes
:	А*
dtype0
~
Output-Layer_2/biasVarHandleOp*
dtype0*
shape:*
_output_shapes
: *$
shared_nameOutput-Layer_2/bias
w
'Output-Layer_2/bias/Read/ReadVariableOpReadVariableOpOutput-Layer_2/bias*
_output_shapes
:*
dtype0
|
training_2/Adam/iterVarHandleOp*
shape: *
dtype0	*%
shared_nametraining_2/Adam/iter*
_output_shapes
: 
u
(training_2/Adam/iter/Read/ReadVariableOpReadVariableOptraining_2/Adam/iter*
dtype0	*
_output_shapes
: 
А
training_2/Adam/beta_1VarHandleOp*
_output_shapes
: *
shape: *
dtype0*'
shared_nametraining_2/Adam/beta_1
y
*training_2/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_1*
dtype0*
_output_shapes
: 
А
training_2/Adam/beta_2VarHandleOp*
dtype0*'
shared_nametraining_2/Adam/beta_2*
_output_shapes
: *
shape: 
y
*training_2/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_2/Adam/beta_2*
dtype0*
_output_shapes
: 
~
training_2/Adam/decayVarHandleOp*&
shared_nametraining_2/Adam/decay*
_output_shapes
: *
dtype0*
shape: 
w
)training_2/Adam/decay/Read/ReadVariableOpReadVariableOptraining_2/Adam/decay*
dtype0*
_output_shapes
: 
О
training_2/Adam/learning_rateVarHandleOp*
dtype0*.
shared_nametraining_2/Adam/learning_rate*
_output_shapes
: *
shape: 
З
1training_2/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_2/Adam/learning_rate*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
shape: *
shared_nametotal*
_output_shapes
: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
й
&training_2/Adam/Hidden-Layer1/kernel/mVarHandleOp*
_output_shapes
: *
shape:	А*7
shared_name(&training_2/Adam/Hidden-Layer1/kernel/m*
dtype0
в
:training_2/Adam/Hidden-Layer1/kernel/m/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer1/kernel/m*
_output_shapes
:	А*
dtype0
б
$training_2/Adam/Hidden-Layer1/bias/mVarHandleOp*
_output_shapes
: *5
shared_name&$training_2/Adam/Hidden-Layer1/bias/m*
shape:А*
dtype0
Ъ
8training_2/Adam/Hidden-Layer1/bias/m/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer1/bias/m*
_output_shapes	
:А*
dtype0
к
&training_2/Adam/Hidden-Layer2/kernel/mVarHandleOp*7
shared_name(&training_2/Adam/Hidden-Layer2/kernel/m*
shape:
АА*
dtype0*
_output_shapes
: 
г
:training_2/Adam/Hidden-Layer2/kernel/m/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer2/kernel/m*
dtype0* 
_output_shapes
:
АА
б
$training_2/Adam/Hidden-Layer2/bias/mVarHandleOp*
dtype0*
_output_shapes
: *5
shared_name&$training_2/Adam/Hidden-Layer2/bias/m*
shape:А
Ъ
8training_2/Adam/Hidden-Layer2/bias/m/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer2/bias/m*
dtype0*
_output_shapes	
:А
к
&training_2/Adam/Hidden-Layer3/kernel/mVarHandleOp*
shape:
АА*
_output_shapes
: *
dtype0*7
shared_name(&training_2/Adam/Hidden-Layer3/kernel/m
г
:training_2/Adam/Hidden-Layer3/kernel/m/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer3/kernel/m*
dtype0* 
_output_shapes
:
АА
б
$training_2/Adam/Hidden-Layer3/bias/mVarHandleOp*5
shared_name&$training_2/Adam/Hidden-Layer3/bias/m*
dtype0*
shape:А*
_output_shapes
: 
Ъ
8training_2/Adam/Hidden-Layer3/bias/m/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer3/bias/m*
_output_shapes	
:А*
dtype0
к
&training_2/Adam/Hidden-Layer4/kernel/mVarHandleOp*
shape:
АА*
_output_shapes
: *
dtype0*7
shared_name(&training_2/Adam/Hidden-Layer4/kernel/m
г
:training_2/Adam/Hidden-Layer4/kernel/m/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer4/kernel/m*
dtype0* 
_output_shapes
:
АА
б
$training_2/Adam/Hidden-Layer4/bias/mVarHandleOp*
_output_shapes
: *5
shared_name&$training_2/Adam/Hidden-Layer4/bias/m*
dtype0*
shape:А
Ъ
8training_2/Adam/Hidden-Layer4/bias/m/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer4/bias/m*
_output_shapes	
:А*
dtype0
л
'training_2/Adam/Output-Layer_2/kernel/mVarHandleOp*
dtype0*
_output_shapes
: *8
shared_name)'training_2/Adam/Output-Layer_2/kernel/m*
shape:	А
д
;training_2/Adam/Output-Layer_2/kernel/m/Read/ReadVariableOpReadVariableOp'training_2/Adam/Output-Layer_2/kernel/m*
_output_shapes
:	А*
dtype0
в
%training_2/Adam/Output-Layer_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*6
shared_name'%training_2/Adam/Output-Layer_2/bias/m*
shape:
Ы
9training_2/Adam/Output-Layer_2/bias/m/Read/ReadVariableOpReadVariableOp%training_2/Adam/Output-Layer_2/bias/m*
dtype0*
_output_shapes
:
й
&training_2/Adam/Hidden-Layer1/kernel/vVarHandleOp*
shape:	А*
dtype0*7
shared_name(&training_2/Adam/Hidden-Layer1/kernel/v*
_output_shapes
: 
в
:training_2/Adam/Hidden-Layer1/kernel/v/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer1/kernel/v*
dtype0*
_output_shapes
:	А
б
$training_2/Adam/Hidden-Layer1/bias/vVarHandleOp*
shape:А*
dtype0*5
shared_name&$training_2/Adam/Hidden-Layer1/bias/v*
_output_shapes
: 
Ъ
8training_2/Adam/Hidden-Layer1/bias/v/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer1/bias/v*
_output_shapes	
:А*
dtype0
к
&training_2/Adam/Hidden-Layer2/kernel/vVarHandleOp*
dtype0*
shape:
АА*7
shared_name(&training_2/Adam/Hidden-Layer2/kernel/v*
_output_shapes
: 
г
:training_2/Adam/Hidden-Layer2/kernel/v/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer2/kernel/v* 
_output_shapes
:
АА*
dtype0
б
$training_2/Adam/Hidden-Layer2/bias/vVarHandleOp*5
shared_name&$training_2/Adam/Hidden-Layer2/bias/v*
shape:А*
_output_shapes
: *
dtype0
Ъ
8training_2/Adam/Hidden-Layer2/bias/v/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer2/bias/v*
_output_shapes	
:А*
dtype0
к
&training_2/Adam/Hidden-Layer3/kernel/vVarHandleOp*7
shared_name(&training_2/Adam/Hidden-Layer3/kernel/v*
dtype0*
_output_shapes
: *
shape:
АА
г
:training_2/Adam/Hidden-Layer3/kernel/v/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer3/kernel/v* 
_output_shapes
:
АА*
dtype0
б
$training_2/Adam/Hidden-Layer3/bias/vVarHandleOp*
dtype0*
_output_shapes
: *5
shared_name&$training_2/Adam/Hidden-Layer3/bias/v*
shape:А
Ъ
8training_2/Adam/Hidden-Layer3/bias/v/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer3/bias/v*
dtype0*
_output_shapes	
:А
к
&training_2/Adam/Hidden-Layer4/kernel/vVarHandleOp*7
shared_name(&training_2/Adam/Hidden-Layer4/kernel/v*
shape:
АА*
_output_shapes
: *
dtype0
г
:training_2/Adam/Hidden-Layer4/kernel/v/Read/ReadVariableOpReadVariableOp&training_2/Adam/Hidden-Layer4/kernel/v*
dtype0* 
_output_shapes
:
АА
б
$training_2/Adam/Hidden-Layer4/bias/vVarHandleOp*
_output_shapes
: *5
shared_name&$training_2/Adam/Hidden-Layer4/bias/v*
shape:А*
dtype0
Ъ
8training_2/Adam/Hidden-Layer4/bias/v/Read/ReadVariableOpReadVariableOp$training_2/Adam/Hidden-Layer4/bias/v*
_output_shapes	
:А*
dtype0
л
'training_2/Adam/Output-Layer_2/kernel/vVarHandleOp*
shape:	А*8
shared_name)'training_2/Adam/Output-Layer_2/kernel/v*
dtype0*
_output_shapes
: 
д
;training_2/Adam/Output-Layer_2/kernel/v/Read/ReadVariableOpReadVariableOp'training_2/Adam/Output-Layer_2/kernel/v*
dtype0*
_output_shapes
:	А
в
%training_2/Adam/Output-Layer_2/bias/vVarHandleOp*
dtype0*
shape:*
_output_shapes
: *6
shared_name'%training_2/Adam/Output-Layer_2/bias/v
Ы
9training_2/Adam/Output-Layer_2/bias/v/Read/ReadVariableOpReadVariableOp%training_2/Adam/Output-Layer_2/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
├?
ConstConst"/device:CPU:0*
_output_shapes
: *■>
valueЇ>Bё> Bъ>
█
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
h

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
h

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
h

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
Ў
9iter

:beta_1

;beta_2
	<decay
=learning_ratemnmompmq#mr$ms)mt*mu3mv4mwvxvyvzv{#v|$v})v~*v3vА4vБ
F
0
1
2
3
#4
$5
)6
*7
38
49
 
F
0
1
2
3
#4
$5
)6
*7
38
49
Ъ
>metrics

trainable_variables

?layers
regularization_losses
@non_trainable_variables
	variables
Alayer_regularization_losses
 
 
 
 
Ъ
Bmetrics
trainable_variables

Clayers
Dnon_trainable_variables
regularization_losses
	variables
Elayer_regularization_losses
 
 
 
Ъ
Fmetrics
trainable_variables

Glayers
Hnon_trainable_variables
regularization_losses
	variables
Ilayer_regularization_losses
`^
VARIABLE_VALUEHidden-Layer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEHidden-Layer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Ъ
Jmetrics
trainable_variables

Klayers
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
`^
VARIABLE_VALUEHidden-Layer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEHidden-Layer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
Ъ
Nmetrics
trainable_variables

Olayers
Pnon_trainable_variables
 regularization_losses
!	variables
Qlayer_regularization_losses
`^
VARIABLE_VALUEHidden-Layer3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEHidden-Layer3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

#0
$1
 

#0
$1
Ъ
Rmetrics
%trainable_variables

Slayers
Tnon_trainable_variables
&regularization_losses
'	variables
Ulayer_regularization_losses
`^
VARIABLE_VALUEHidden-Layer4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEHidden-Layer4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

)0
*1
 

)0
*1
Ъ
Vmetrics
+trainable_variables

Wlayers
Xnon_trainable_variables
,regularization_losses
-	variables
Ylayer_regularization_losses
 
 
 
Ъ
Zmetrics
/trainable_variables

[layers
\non_trainable_variables
0regularization_losses
1	variables
]layer_regularization_losses
a_
VARIABLE_VALUEOutput-Layer_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEOutput-Layer_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
Ъ
^metrics
5trainable_variables

_layers
`non_trainable_variables
6regularization_losses
7	variables
alayer_regularization_losses
SQ
VARIABLE_VALUEtraining_2/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_2/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_2/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_2/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

b0
1
0
1
2
3
4
5
6
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
 
 
 
 
 
 
 
 
x
	ctotal
	dcount
e
_fn_kwargs
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

c0
d1
Ъ
jmetrics
ftrainable_variables

klayers
lnon_trainable_variables
gregularization_losses
h	variables
mlayer_regularization_losses
 
 

c0
d1
 
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE'training_2/Adam/Output-Layer_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE%training_2/Adam/Output-Layer_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE&training_2/Adam/Hidden-Layer4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE$training_2/Adam/Hidden-Layer4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
РН
VARIABLE_VALUE'training_2/Adam/Output-Layer_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
МЙ
VARIABLE_VALUE%training_2/Adam/Output-Layer_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
В
serving_default_flatten_2_inputPlaceholder*
shape:         *'
_output_shapes
:         *
dtype0
Г
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_2_inputHidden-Layer1/kernelHidden-Layer1/biasHidden-Layer2/kernelHidden-Layer2/biasHidden-Layer3/kernelHidden-Layer3/biasHidden-Layer4/kernelHidden-Layer4/biasOutput-Layer_2/kernelOutput-Layer_2/bias*'
_output_shapes
:         *.
f)R'
%__inference_signature_wrapper_6119828**
config_proto

GPU 

CPU2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-6120173*
Tout
2
O
saver_filenamePlaceholder*
_output_shapes
: *
shape: *
dtype0
 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(Hidden-Layer1/kernel/Read/ReadVariableOp&Hidden-Layer1/bias/Read/ReadVariableOp(Hidden-Layer2/kernel/Read/ReadVariableOp&Hidden-Layer2/bias/Read/ReadVariableOp(Hidden-Layer3/kernel/Read/ReadVariableOp&Hidden-Layer3/bias/Read/ReadVariableOp(Hidden-Layer4/kernel/Read/ReadVariableOp&Hidden-Layer4/bias/Read/ReadVariableOp)Output-Layer_2/kernel/Read/ReadVariableOp'Output-Layer_2/bias/Read/ReadVariableOp(training_2/Adam/iter/Read/ReadVariableOp*training_2/Adam/beta_1/Read/ReadVariableOp*training_2/Adam/beta_2/Read/ReadVariableOp)training_2/Adam/decay/Read/ReadVariableOp1training_2/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp:training_2/Adam/Hidden-Layer1/kernel/m/Read/ReadVariableOp8training_2/Adam/Hidden-Layer1/bias/m/Read/ReadVariableOp:training_2/Adam/Hidden-Layer2/kernel/m/Read/ReadVariableOp8training_2/Adam/Hidden-Layer2/bias/m/Read/ReadVariableOp:training_2/Adam/Hidden-Layer3/kernel/m/Read/ReadVariableOp8training_2/Adam/Hidden-Layer3/bias/m/Read/ReadVariableOp:training_2/Adam/Hidden-Layer4/kernel/m/Read/ReadVariableOp8training_2/Adam/Hidden-Layer4/bias/m/Read/ReadVariableOp;training_2/Adam/Output-Layer_2/kernel/m/Read/ReadVariableOp9training_2/Adam/Output-Layer_2/bias/m/Read/ReadVariableOp:training_2/Adam/Hidden-Layer1/kernel/v/Read/ReadVariableOp8training_2/Adam/Hidden-Layer1/bias/v/Read/ReadVariableOp:training_2/Adam/Hidden-Layer2/kernel/v/Read/ReadVariableOp8training_2/Adam/Hidden-Layer2/bias/v/Read/ReadVariableOp:training_2/Adam/Hidden-Layer3/kernel/v/Read/ReadVariableOp8training_2/Adam/Hidden-Layer3/bias/v/Read/ReadVariableOp:training_2/Adam/Hidden-Layer4/kernel/v/Read/ReadVariableOp8training_2/Adam/Hidden-Layer4/bias/v/Read/ReadVariableOp;training_2/Adam/Output-Layer_2/kernel/v/Read/ReadVariableOp9training_2/Adam/Output-Layer_2/bias/v/Read/ReadVariableOpConst*
Tout
2*.
_gradient_op_typePartitionedCall-6120232**
config_proto

GPU 

CPU2J 8*)
f$R"
 __inference__traced_save_6120231*
_output_shapes
: *2
Tin+
)2'	
Ц
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden-Layer1/kernelHidden-Layer1/biasHidden-Layer2/kernelHidden-Layer2/biasHidden-Layer3/kernelHidden-Layer3/biasHidden-Layer4/kernelHidden-Layer4/biasOutput-Layer_2/kernelOutput-Layer_2/biastraining_2/Adam/itertraining_2/Adam/beta_1training_2/Adam/beta_2training_2/Adam/decaytraining_2/Adam/learning_ratetotalcount&training_2/Adam/Hidden-Layer1/kernel/m$training_2/Adam/Hidden-Layer1/bias/m&training_2/Adam/Hidden-Layer2/kernel/m$training_2/Adam/Hidden-Layer2/bias/m&training_2/Adam/Hidden-Layer3/kernel/m$training_2/Adam/Hidden-Layer3/bias/m&training_2/Adam/Hidden-Layer4/kernel/m$training_2/Adam/Hidden-Layer4/bias/m'training_2/Adam/Output-Layer_2/kernel/m%training_2/Adam/Output-Layer_2/bias/m&training_2/Adam/Hidden-Layer1/kernel/v$training_2/Adam/Hidden-Layer1/bias/v&training_2/Adam/Hidden-Layer2/kernel/v$training_2/Adam/Hidden-Layer2/bias/v&training_2/Adam/Hidden-Layer3/kernel/v$training_2/Adam/Hidden-Layer3/bias/v&training_2/Adam/Hidden-Layer4/kernel/v$training_2/Adam/Hidden-Layer4/bias/v'training_2/Adam/Output-Layer_2/kernel/v%training_2/Adam/Output-Layer_2/bias/v**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6120356*
_output_shapes
: *
Tout
2*1
Tin*
(2&*,
f'R%
#__inference__traced_restore_6120355Е░
Й
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120067

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:         А*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:         А*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
лA
─	
"__inference__wrapped_model_6119497
flatten_2_input=
9sequential_2_hidden_layer1_matmul_readvariableop_resource>
:sequential_2_hidden_layer1_biasadd_readvariableop_resource=
9sequential_2_hidden_layer2_matmul_readvariableop_resource>
:sequential_2_hidden_layer2_biasadd_readvariableop_resource=
9sequential_2_hidden_layer3_matmul_readvariableop_resource>
:sequential_2_hidden_layer3_biasadd_readvariableop_resource=
9sequential_2_hidden_layer4_matmul_readvariableop_resource>
:sequential_2_hidden_layer4_biasadd_readvariableop_resource<
8sequential_2_output_layer_matmul_readvariableop_resource=
9sequential_2_output_layer_biasadd_readvariableop_resource
identityИв1sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOpв0sequential_2/Hidden-Layer1/MatMul/ReadVariableOpв1sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOpв0sequential_2/Hidden-Layer2/MatMul/ReadVariableOpв1sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOpв0sequential_2/Hidden-Layer3/MatMul/ReadVariableOpв1sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOpв0sequential_2/Hidden-Layer4/MatMul/ReadVariableOpв0sequential_2/Output-Layer/BiasAdd/ReadVariableOpв/sequential_2/Output-Layer/MatMul/ReadVariableOpu
$sequential_2/flatten_2/Reshape/shapeConst*
valueB"       *
_output_shapes
:*
dtype0Ы
sequential_2/flatten_2/ReshapeReshapeflatten_2_input-sequential_2/flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:         ┘
0sequential_2/Hidden-Layer1/MatMul/ReadVariableOpReadVariableOp9sequential_2_hidden_layer1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0┴
!sequential_2/Hidden-Layer1/MatMulMatMul'sequential_2/flatten_2/Reshape:output:08sequential_2/Hidden-Layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╫
1sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_hidden_layer1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0╚
"sequential_2/Hidden-Layer1/BiasAddBiasAdd+sequential_2/Hidden-Layer1/MatMul:product:09sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
sequential_2/Hidden-Layer1/TanhTanh+sequential_2/Hidden-Layer1/BiasAdd:output:0*(
_output_shapes
:         А*
T0┌
0sequential_2/Hidden-Layer2/MatMul/ReadVariableOpReadVariableOp9sequential_2_hidden_layer2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
АА╜
!sequential_2/Hidden-Layer2/MatMulMatMul#sequential_2/Hidden-Layer1/Tanh:y:08sequential_2/Hidden-Layer2/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╫
1sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_hidden_layer2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:А╚
"sequential_2/Hidden-Layer2/BiasAddBiasAdd+sequential_2/Hidden-Layer2/MatMul:product:09sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0З
sequential_2/Hidden-Layer2/TanhTanh+sequential_2/Hidden-Layer2/BiasAdd:output:0*(
_output_shapes
:         А*
T0┌
0sequential_2/Hidden-Layer3/MatMul/ReadVariableOpReadVariableOp9sequential_2_hidden_layer3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
АА╜
!sequential_2/Hidden-Layer3/MatMulMatMul#sequential_2/Hidden-Layer2/Tanh:y:08sequential_2/Hidden-Layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╫
1sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_hidden_layer3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0╚
"sequential_2/Hidden-Layer3/BiasAddBiasAdd+sequential_2/Hidden-Layer3/MatMul:product:09sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
sequential_2/Hidden-Layer3/TanhTanh+sequential_2/Hidden-Layer3/BiasAdd:output:0*(
_output_shapes
:         А*
T0┌
0sequential_2/Hidden-Layer4/MatMul/ReadVariableOpReadVariableOp9sequential_2_hidden_layer4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
АА╜
!sequential_2/Hidden-Layer4/MatMulMatMul#sequential_2/Hidden-Layer3/Tanh:y:08sequential_2/Hidden-Layer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╫
1sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOpReadVariableOp:sequential_2_hidden_layer4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0╚
"sequential_2/Hidden-Layer4/BiasAddBiasAdd+sequential_2/Hidden-Layer4/MatMul:product:09sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЗ
sequential_2/Hidden-Layer4/TanhTanh+sequential_2/Hidden-Layer4/BiasAdd:output:0*
T0*(
_output_shapes
:         АГ
sequential_2/dropout_2/IdentityIdentity#sequential_2/Hidden-Layer4/Tanh:y:0*(
_output_shapes
:         А*
T0╫
/sequential_2/Output-Layer/MatMul/ReadVariableOpReadVariableOp8sequential_2_output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	А┐
 sequential_2/Output-Layer/MatMulMatMul(sequential_2/dropout_2/Identity:output:07sequential_2/Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╘
0sequential_2/Output-Layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_2_output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0─
!sequential_2/Output-Layer/BiasAddBiasAdd*sequential_2/Output-Layer/MatMul:product:08sequential_2/Output-Layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0К
!sequential_2/Output-Layer/SigmoidSigmoid*sequential_2/Output-Layer/BiasAdd:output:0*'
_output_shapes
:         *
T0ю
IdentityIdentity%sequential_2/Output-Layer/Sigmoid:y:02^sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOp1^sequential_2/Hidden-Layer1/MatMul/ReadVariableOp2^sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOp1^sequential_2/Hidden-Layer2/MatMul/ReadVariableOp2^sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOp1^sequential_2/Hidden-Layer3/MatMul/ReadVariableOp2^sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOp1^sequential_2/Hidden-Layer4/MatMul/ReadVariableOp1^sequential_2/Output-Layer/BiasAdd/ReadVariableOp0^sequential_2/Output-Layer/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2d
0sequential_2/Output-Layer/BiasAdd/ReadVariableOp0sequential_2/Output-Layer/BiasAdd/ReadVariableOp2d
0sequential_2/Hidden-Layer1/MatMul/ReadVariableOp0sequential_2/Hidden-Layer1/MatMul/ReadVariableOp2d
0sequential_2/Hidden-Layer3/MatMul/ReadVariableOp0sequential_2/Hidden-Layer3/MatMul/ReadVariableOp2f
1sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOp1sequential_2/Hidden-Layer4/BiasAdd/ReadVariableOp2b
/sequential_2/Output-Layer/MatMul/ReadVariableOp/sequential_2/Output-Layer/MatMul/ReadVariableOp2f
1sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOp1sequential_2/Hidden-Layer3/BiasAdd/ReadVariableOp2f
1sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOp1sequential_2/Hidden-Layer2/BiasAdd/ReadVariableOp2d
0sequential_2/Hidden-Layer2/MatMul/ReadVariableOp0sequential_2/Hidden-Layer2/MatMul/ReadVariableOp2d
0sequential_2/Hidden-Layer4/MatMul/ReadVariableOp0sequential_2/Hidden-Layer4/MatMul/ReadVariableOp2f
1sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOp1sequential_2/Hidden-Layer1/BiasAdd/ReadVariableOp:/ +
)
_user_specified_nameflatten_2_input: : : : : : : : :	 :
 
°
┘
.__inference_sequential_2_layer_call_fn_6119768
flatten_2_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tout
2*.
_gradient_op_typePartitionedCall-6119755**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119754В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameflatten_2_input: : : : : : : : :	 :
 
╘	
у
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6120017

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0б
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
TanhTanhBiasAdd:output:0*(
_output_shapes
:         А*
T0В
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : :& "
 
_user_specified_nameinputs
ш
░
/__inference_Hidden-Layer2_layer_call_fn_6120006

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6119567*
Tin
2*S
fNRL
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:         АГ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
║O
ь
 __inference__traced_save_6120231
file_prefix3
/savev2_hidden_layer1_kernel_read_readvariableop1
-savev2_hidden_layer1_bias_read_readvariableop3
/savev2_hidden_layer2_kernel_read_readvariableop1
-savev2_hidden_layer2_bias_read_readvariableop3
/savev2_hidden_layer3_kernel_read_readvariableop1
-savev2_hidden_layer3_bias_read_readvariableop3
/savev2_hidden_layer4_kernel_read_readvariableop1
-savev2_hidden_layer4_bias_read_readvariableop4
0savev2_output_layer_2_kernel_read_readvariableop2
.savev2_output_layer_2_bias_read_readvariableop3
/savev2_training_2_adam_iter_read_readvariableop	5
1savev2_training_2_adam_beta_1_read_readvariableop5
1savev2_training_2_adam_beta_2_read_readvariableop4
0savev2_training_2_adam_decay_read_readvariableop<
8savev2_training_2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopE
Asavev2_training_2_adam_hidden_layer1_kernel_m_read_readvariableopC
?savev2_training_2_adam_hidden_layer1_bias_m_read_readvariableopE
Asavev2_training_2_adam_hidden_layer2_kernel_m_read_readvariableopC
?savev2_training_2_adam_hidden_layer2_bias_m_read_readvariableopE
Asavev2_training_2_adam_hidden_layer3_kernel_m_read_readvariableopC
?savev2_training_2_adam_hidden_layer3_bias_m_read_readvariableopE
Asavev2_training_2_adam_hidden_layer4_kernel_m_read_readvariableopC
?savev2_training_2_adam_hidden_layer4_bias_m_read_readvariableopF
Bsavev2_training_2_adam_output_layer_2_kernel_m_read_readvariableopD
@savev2_training_2_adam_output_layer_2_bias_m_read_readvariableopE
Asavev2_training_2_adam_hidden_layer1_kernel_v_read_readvariableopC
?savev2_training_2_adam_hidden_layer1_bias_v_read_readvariableopE
Asavev2_training_2_adam_hidden_layer2_kernel_v_read_readvariableopC
?savev2_training_2_adam_hidden_layer2_bias_v_read_readvariableopE
Asavev2_training_2_adam_hidden_layer3_kernel_v_read_readvariableopC
?savev2_training_2_adam_hidden_layer3_bias_v_read_readvariableopE
Asavev2_training_2_adam_hidden_layer4_kernel_v_read_readvariableopC
?savev2_training_2_adam_hidden_layer4_bias_v_read_readvariableopF
Bsavev2_training_2_adam_output_layer_2_kernel_v_read_readvariableopD
@savev2_training_2_adam_output_layer_2_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *<
value3B1 B+_temp_6e84793449ec4d6fa9152bc370520ff7/part*
dtype0s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ▀
SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*И
value■B√%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
_output_shapes
:%╖
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0Ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_hidden_layer1_kernel_read_readvariableop-savev2_hidden_layer1_bias_read_readvariableop/savev2_hidden_layer2_kernel_read_readvariableop-savev2_hidden_layer2_bias_read_readvariableop/savev2_hidden_layer3_kernel_read_readvariableop-savev2_hidden_layer3_bias_read_readvariableop/savev2_hidden_layer4_kernel_read_readvariableop-savev2_hidden_layer4_bias_read_readvariableop0savev2_output_layer_2_kernel_read_readvariableop.savev2_output_layer_2_bias_read_readvariableop/savev2_training_2_adam_iter_read_readvariableop1savev2_training_2_adam_beta_1_read_readvariableop1savev2_training_2_adam_beta_2_read_readvariableop0savev2_training_2_adam_decay_read_readvariableop8savev2_training_2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopAsavev2_training_2_adam_hidden_layer1_kernel_m_read_readvariableop?savev2_training_2_adam_hidden_layer1_bias_m_read_readvariableopAsavev2_training_2_adam_hidden_layer2_kernel_m_read_readvariableop?savev2_training_2_adam_hidden_layer2_bias_m_read_readvariableopAsavev2_training_2_adam_hidden_layer3_kernel_m_read_readvariableop?savev2_training_2_adam_hidden_layer3_bias_m_read_readvariableopAsavev2_training_2_adam_hidden_layer4_kernel_m_read_readvariableop?savev2_training_2_adam_hidden_layer4_bias_m_read_readvariableopBsavev2_training_2_adam_output_layer_2_kernel_m_read_readvariableop@savev2_training_2_adam_output_layer_2_bias_m_read_readvariableopAsavev2_training_2_adam_hidden_layer1_kernel_v_read_readvariableop?savev2_training_2_adam_hidden_layer1_bias_v_read_readvariableopAsavev2_training_2_adam_hidden_layer2_kernel_v_read_readvariableop?savev2_training_2_adam_hidden_layer2_bias_v_read_readvariableopAsavev2_training_2_adam_hidden_layer3_kernel_v_read_readvariableop?savev2_training_2_adam_hidden_layer3_bias_v_read_readvariableopAsavev2_training_2_adam_hidden_layer4_kernel_v_read_readvariableop?savev2_training_2_adam_hidden_layer4_bias_v_read_readvariableopBsavev2_training_2_adam_output_layer_2_kernel_v_read_readvariableop@savev2_training_2_adam_output_layer_2_bias_v_read_readvariableop"/device:CPU:0*3
dtypes)
'2%	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
valueB
B *
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
dtypes
2*
_output_shapes
 ╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0Ц
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
_output_shapes
: *
T0s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*╗
_input_shapesй
ж: :	А:А:
АА:А:
АА:А:
АА:А:	А:: : : : : : : :	А:А:
АА:А:
АА:А:
АА:А:	А::	А:А:
АА:А:
АА:А:
АА:А:	А:: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2: : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 
╘	
у
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6120035

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         АВ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╓	
т
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6120088

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
╠5
▐
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119929

inputs0
,hidden_layer1_matmul_readvariableop_resource1
-hidden_layer1_biasadd_readvariableop_resource0
,hidden_layer2_matmul_readvariableop_resource1
-hidden_layer2_biasadd_readvariableop_resource0
,hidden_layer3_matmul_readvariableop_resource1
-hidden_layer3_biasadd_readvariableop_resource0
,hidden_layer4_matmul_readvariableop_resource1
-hidden_layer4_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИв$Hidden-Layer1/BiasAdd/ReadVariableOpв#Hidden-Layer1/MatMul/ReadVariableOpв$Hidden-Layer2/BiasAdd/ReadVariableOpв#Hidden-Layer2/MatMul/ReadVariableOpв$Hidden-Layer3/BiasAdd/ReadVariableOpв#Hidden-Layer3/MatMul/ReadVariableOpв$Hidden-Layer4/BiasAdd/ReadVariableOpв#Hidden-Layer4/MatMul/ReadVariableOpв#Output-Layer/BiasAdd/ReadVariableOpв"Output-Layer/MatMul/ReadVariableOph
flatten_2/Reshape/shapeConst*
_output_shapes
:*
valueB"       *
dtype0x
flatten_2/ReshapeReshapeinputs flatten_2/Reshape/shape:output:0*
T0*'
_output_shapes
:         ┐
#Hidden-Layer1/MatMul/ReadVariableOpReadVariableOp,hidden_layer1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АЪ
Hidden-Layer1/MatMulMatMulflatten_2/Reshape:output:0+Hidden-Layer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╜
$Hidden-Layer1/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0б
Hidden-Layer1/BiasAddBiasAddHidden-Layer1/MatMul:product:0,Hidden-Layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer1/TanhTanhHidden-Layer1/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer2/MatMul/ReadVariableOpReadVariableOp,hidden_layer2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
АА*
dtype0Ц
Hidden-Layer2/MatMulMatMulHidden-Layer1/Tanh:y:0+Hidden-Layer2/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╜
$Hidden-Layer2/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аб
Hidden-Layer2/BiasAddBiasAddHidden-Layer2/MatMul:product:0,Hidden-Layer2/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0m
Hidden-Layer2/TanhTanhHidden-Layer2/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer3/MatMul/ReadVariableOpReadVariableOp,hidden_layer3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААЦ
Hidden-Layer3/MatMulMatMulHidden-Layer2/Tanh:y:0+Hidden-Layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╜
$Hidden-Layer3/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аб
Hidden-Layer3/BiasAddBiasAddHidden-Layer3/MatMul:product:0,Hidden-Layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer3/TanhTanhHidden-Layer3/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer4/MatMul/ReadVariableOpReadVariableOp,hidden_layer4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААЦ
Hidden-Layer4/MatMulMatMulHidden-Layer3/Tanh:y:0+Hidden-Layer4/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╜
$Hidden-Layer4/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0б
Hidden-Layer4/BiasAddBiasAddHidden-Layer4/MatMul:product:0,Hidden-Layer4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0m
Hidden-Layer4/TanhTanhHidden-Layer4/BiasAdd:output:0*
T0*(
_output_shapes
:         Аi
dropout_2/IdentityIdentityHidden-Layer4/Tanh:y:0*(
_output_shapes
:         А*
T0╜
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0Ш
Output-Layer/MatMulMatMuldropout_2/Identity:output:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ║
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0Э
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         p
Output-Layer/SigmoidSigmoidOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:         ▀
IdentityIdentityOutput-Layer/Sigmoid:y:0%^Hidden-Layer1/BiasAdd/ReadVariableOp$^Hidden-Layer1/MatMul/ReadVariableOp%^Hidden-Layer2/BiasAdd/ReadVariableOp$^Hidden-Layer2/MatMul/ReadVariableOp%^Hidden-Layer3/BiasAdd/ReadVariableOp$^Hidden-Layer3/MatMul/ReadVariableOp%^Hidden-Layer4/BiasAdd/ReadVariableOp$^Hidden-Layer4/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2L
$Hidden-Layer4/BiasAdd/ReadVariableOp$Hidden-Layer4/BiasAdd/ReadVariableOp2J
#Hidden-Layer2/MatMul/ReadVariableOp#Hidden-Layer2/MatMul/ReadVariableOp2L
$Hidden-Layer3/BiasAdd/ReadVariableOp$Hidden-Layer3/BiasAdd/ReadVariableOp2J
#Hidden-Layer4/MatMul/ReadVariableOp#Hidden-Layer4/MatMul/ReadVariableOp2L
$Hidden-Layer2/BiasAdd/ReadVariableOp$Hidden-Layer2/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp2L
$Hidden-Layer1/BiasAdd/ReadVariableOp$Hidden-Layer1/BiasAdd/ReadVariableOp2J
#Hidden-Layer1/MatMul/ReadVariableOp#Hidden-Layer1/MatMul/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2J
#Hidden-Layer3/MatMul/ReadVariableOp#Hidden-Layer3/MatMul/ReadVariableOp: : : : : : : : :	 :
 :& "
 
_user_specified_nameinputs
С$
б
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119793

inputs0
,hidden_layer1_statefulpartitionedcall_args_10
,hidden_layer1_statefulpartitionedcall_args_20
,hidden_layer2_statefulpartitionedcall_args_10
,hidden_layer2_statefulpartitionedcall_args_20
,hidden_layer3_statefulpartitionedcall_args_10
,hidden_layer3_statefulpartitionedcall_args_20
,hidden_layer4_statefulpartitionedcall_args_10
,hidden_layer4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИв%Hidden-Layer1/StatefulPartitionedCallв%Hidden-Layer2/StatefulPartitionedCallв%Hidden-Layer3/StatefulPartitionedCallв%Hidden-Layer4/StatefulPartitionedCallв$Output-Layer/StatefulPartitionedCallз
flatten_2/PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-6119515*
Tin
2*O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509║
%Hidden-Layer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0,hidden_layer1_statefulpartitionedcall_args_1,hidden_layer1_statefulpartitionedcall_args_2*(
_output_shapes
:         А*.
_gradient_op_typePartitionedCall-6119539*
Tin
2*S
fNRL
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533**
config_proto

GPU 

CPU2J 8*
Tout
2╞
%Hidden-Layer2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer1/StatefulPartitionedCall:output:0,hidden_layer2_statefulpartitionedcall_args_1,hidden_layer2_statefulpartitionedcall_args_2*
Tout
2*S
fNRL
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561*.
_gradient_op_typePartitionedCall-6119567*(
_output_shapes
:         А*
Tin
2**
config_proto

GPU 

CPU2J 8╞
%Hidden-Layer3/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer2/StatefulPartitionedCall:output:0,hidden_layer3_statefulpartitionedcall_args_1,hidden_layer3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6119595*S
fNRL
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589*(
_output_shapes
:         А*
Tin
2*
Tout
2╞
%Hidden-Layer4/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer3/StatefulPartitionedCall:output:0,hidden_layer4_statefulpartitionedcall_args_1,hidden_layer4_statefulpartitionedcall_args_2*
Tout
2*S
fNRL
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617*.
_gradient_op_typePartitionedCall-6119623**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:         А*
Tin
2╨
dropout_2/PartitionedCallPartitionedCall.Hidden-Layer4/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119661*
Tin
2*
Tout
2*.
_gradient_op_typePartitionedCall-6119673*(
_output_shapes
:         А╡
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6119695*'
_output_shapes
:         *R
fMRK
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689*
Tin
2*
Tout
2╝
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0&^Hidden-Layer1/StatefulPartitionedCall&^Hidden-Layer2/StatefulPartitionedCall&^Hidden-Layer3/StatefulPartitionedCall&^Hidden-Layer4/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall2N
%Hidden-Layer1/StatefulPartitionedCall%Hidden-Layer1/StatefulPartitionedCall2N
%Hidden-Layer2/StatefulPartitionedCall%Hidden-Layer2/StatefulPartitionedCall2N
%Hidden-Layer3/StatefulPartitionedCall%Hidden-Layer3/StatefulPartitionedCall2N
%Hidden-Layer4/StatefulPartitionedCall%Hidden-Layer4/StatefulPartitionedCall: : : : : : : :	 :
 :& "
 
_user_specified_nameinputs: 
м$
к
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119730
flatten_2_input0
,hidden_layer1_statefulpartitionedcall_args_10
,hidden_layer1_statefulpartitionedcall_args_20
,hidden_layer2_statefulpartitionedcall_args_10
,hidden_layer2_statefulpartitionedcall_args_20
,hidden_layer3_statefulpartitionedcall_args_10
,hidden_layer3_statefulpartitionedcall_args_20
,hidden_layer4_statefulpartitionedcall_args_10
,hidden_layer4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИв%Hidden-Layer1/StatefulPartitionedCallв%Hidden-Layer2/StatefulPartitionedCallв%Hidden-Layer3/StatefulPartitionedCallв%Hidden-Layer4/StatefulPartitionedCallв$Output-Layer/StatefulPartitionedCall░
flatten_2/PartitionedCallPartitionedCallflatten_2_input**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-6119515*O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509*
Tin
2║
%Hidden-Layer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0,hidden_layer1_statefulpartitionedcall_args_1,hidden_layer1_statefulpartitionedcall_args_2*
Tin
2**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:         А*S
fNRL
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533*.
_gradient_op_typePartitionedCall-6119539╞
%Hidden-Layer2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer1/StatefulPartitionedCall:output:0,hidden_layer2_statefulpartitionedcall_args_1,hidden_layer2_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*S
fNRL
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561*
Tin
2*(
_output_shapes
:         А*.
_gradient_op_typePartitionedCall-6119567╞
%Hidden-Layer3/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer2/StatefulPartitionedCall:output:0,hidden_layer3_statefulpartitionedcall_args_1,hidden_layer3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*S
fNRL
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589*
Tin
2*(
_output_shapes
:         А*.
_gradient_op_typePartitionedCall-6119595╞
%Hidden-Layer4/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer3/StatefulPartitionedCall:output:0,hidden_layer4_statefulpartitionedcall_args_1,hidden_layer4_statefulpartitionedcall_args_2*S
fNRL
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617*
Tout
2*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-6119623╨
dropout_2/PartitionedCallPartitionedCall.Hidden-Layer4/StatefulPartitionedCall:output:0*
Tin
2*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119661*.
_gradient_op_typePartitionedCall-6119673*(
_output_shapes
:         А*
Tout
2**
config_proto

GPU 

CPU2J 8╡
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*'
_output_shapes
:         *R
fMRK
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6119695*
Tout
2*
Tin
2╝
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0&^Hidden-Layer1/StatefulPartitionedCall&^Hidden-Layer2/StatefulPartitionedCall&^Hidden-Layer3/StatefulPartitionedCall&^Hidden-Layer4/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2N
%Hidden-Layer1/StatefulPartitionedCall%Hidden-Layer1/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall2N
%Hidden-Layer2/StatefulPartitionedCall%Hidden-Layer2/StatefulPartitionedCall2N
%Hidden-Layer3/StatefulPartitionedCall%Hidden-Layer3/StatefulPartitionedCall2N
%Hidden-Layer4/StatefulPartitionedCall%Hidden-Layer4/StatefulPartitionedCall:/ +
)
_user_specified_nameflatten_2_input: : : : : : : : :	 :
 
ш
░
/__inference_Hidden-Layer4_layer_call_fn_6120042

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2*.
_gradient_op_typePartitionedCall-6119623*S
fNRL
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:         А*
Tin
2Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
│
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120062

inputs
identityИQ
dropout/rateConst*
_output_shapes
: *
valueB
 *═╠L>*
dtype0C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*(
_output_shapes
:         А*
T0М
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: г
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:         А*
T0Х
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         АR
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0К
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:         А*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:         А*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         А*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:         А*
T0Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╥	
у
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	Аj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
TanhTanhBiasAdd:output:0*(
_output_shapes
:         А*
T0В
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*.
_input_shapes
:         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
║
G
+__inference_flatten_2_layer_call_fn_6119970

inputs
identityЭ
PartitionedCallPartitionedCallinputs*'
_output_shapes
:         *
Tin
2*O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509**
config_proto

GPU 

CPU2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-6119515`
IdentityIdentityPartitionedCall:output:0*'
_output_shapes
:         *
T0"
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
ф
п
.__inference_Output-Layer_layer_call_fn_6120095

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6119695**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:         *R
fMRK
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689*
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
┴
d
+__inference_dropout_2_layer_call_fn_6120072

inputs
identityИвStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*.
_gradient_op_typePartitionedCall-6119665*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119654*(
_output_shapes
:         А*
Tout
2Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
╘	
у
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119999

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0б
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         АВ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╚
╨
%__inference_signature_wrapper_6119828
flatten_2_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*
Tin
2*.
_gradient_op_typePartitionedCall-6119815*
Tout
2*+
f&R$
"__inference__wrapped_model_6119497**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:/ +
)
_user_specified_nameflatten_2_input: : : : : : : : :	 :
 
╣%
┼
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119754

inputs0
,hidden_layer1_statefulpartitionedcall_args_10
,hidden_layer1_statefulpartitionedcall_args_20
,hidden_layer2_statefulpartitionedcall_args_10
,hidden_layer2_statefulpartitionedcall_args_20
,hidden_layer3_statefulpartitionedcall_args_10
,hidden_layer3_statefulpartitionedcall_args_20
,hidden_layer4_statefulpartitionedcall_args_10
,hidden_layer4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИв%Hidden-Layer1/StatefulPartitionedCallв%Hidden-Layer2/StatefulPartitionedCallв%Hidden-Layer3/StatefulPartitionedCallв%Hidden-Layer4/StatefulPartitionedCallв$Output-Layer/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCallз
flatten_2/PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-6119515*
Tin
2*O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509*
Tout
2*'
_output_shapes
:         **
config_proto

GPU 

CPU2J 8║
%Hidden-Layer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0,hidden_layer1_statefulpartitionedcall_args_1,hidden_layer1_statefulpartitionedcall_args_2*S
fNRL
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6119539*(
_output_shapes
:         А*
Tout
2*
Tin
2╞
%Hidden-Layer2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer1/StatefulPartitionedCall:output:0,hidden_layer2_statefulpartitionedcall_args_1,hidden_layer2_statefulpartitionedcall_args_2*
Tout
2*
Tin
2*S
fNRL
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561*(
_output_shapes
:         А*.
_gradient_op_typePartitionedCall-6119567**
config_proto

GPU 

CPU2J 8╞
%Hidden-Layer3/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer2/StatefulPartitionedCall:output:0,hidden_layer3_statefulpartitionedcall_args_1,hidden_layer3_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*.
_gradient_op_typePartitionedCall-6119595*
Tout
2*S
fNRL
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589*
Tin
2*(
_output_shapes
:         А╞
%Hidden-Layer4/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer3/StatefulPartitionedCall:output:0,hidden_layer4_statefulpartitionedcall_args_1,hidden_layer4_statefulpartitionedcall_args_2*(
_output_shapes
:         А*
Tout
2*.
_gradient_op_typePartitionedCall-6119623**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617*
Tin
2р
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer4/StatefulPartitionedCall:output:0*
Tin
2*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119654*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*
Tout
2*.
_gradient_op_typePartitionedCall-6119665╜
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         *R
fMRK
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689*.
_gradient_op_typePartitionedCall-6119695*
Tout
2р
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0&^Hidden-Layer1/StatefulPartitionedCall&^Hidden-Layer2/StatefulPartitionedCall&^Hidden-Layer3/StatefulPartitionedCall&^Hidden-Layer4/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2N
%Hidden-Layer2/StatefulPartitionedCall%Hidden-Layer2/StatefulPartitionedCall2N
%Hidden-Layer3/StatefulPartitionedCall%Hidden-Layer3/StatefulPartitionedCall2N
%Hidden-Layer4/StatefulPartitionedCall%Hidden-Layer4/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall2N
%Hidden-Layer1/StatefulPartitionedCall%Hidden-Layer1/StatefulPartitionedCall:	 :
 :& "
 
_user_specified_nameinputs: : : : : : : : 
╘%
╬
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119707
flatten_2_input0
,hidden_layer1_statefulpartitionedcall_args_10
,hidden_layer1_statefulpartitionedcall_args_20
,hidden_layer2_statefulpartitionedcall_args_10
,hidden_layer2_statefulpartitionedcall_args_20
,hidden_layer3_statefulpartitionedcall_args_10
,hidden_layer3_statefulpartitionedcall_args_20
,hidden_layer4_statefulpartitionedcall_args_10
,hidden_layer4_statefulpartitionedcall_args_2/
+output_layer_statefulpartitionedcall_args_1/
+output_layer_statefulpartitionedcall_args_2
identityИв%Hidden-Layer1/StatefulPartitionedCallв%Hidden-Layer2/StatefulPartitionedCallв%Hidden-Layer3/StatefulPartitionedCallв%Hidden-Layer4/StatefulPartitionedCallв$Output-Layer/StatefulPartitionedCallв!dropout_2/StatefulPartitionedCall░
flatten_2/PartitionedCallPartitionedCallflatten_2_input*
Tout
2*O
fJRH
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509*.
_gradient_op_typePartitionedCall-6119515*
Tin
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         ║
%Hidden-Layer1/StatefulPartitionedCallStatefulPartitionedCall"flatten_2/PartitionedCall:output:0,hidden_layer1_statefulpartitionedcall_args_1,hidden_layer1_statefulpartitionedcall_args_2*S
fNRL
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:         А*
Tin
2*
Tout
2*.
_gradient_op_typePartitionedCall-6119539╞
%Hidden-Layer2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer1/StatefulPartitionedCall:output:0,hidden_layer2_statefulpartitionedcall_args_1,hidden_layer2_statefulpartitionedcall_args_2*
Tout
2*
Tin
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:         А*S
fNRL
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561*.
_gradient_op_typePartitionedCall-6119567╞
%Hidden-Layer3/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer2/StatefulPartitionedCall:output:0,hidden_layer3_statefulpartitionedcall_args_1,hidden_layer3_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6119595*S
fNRL
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:         А*
Tin
2╞
%Hidden-Layer4/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer3/StatefulPartitionedCall:output:0,hidden_layer4_statefulpartitionedcall_args_1,hidden_layer4_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tout
2*(
_output_shapes
:         А*S
fNRL
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617*.
_gradient_op_typePartitionedCall-6119623*
Tin
2р
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall.Hidden-Layer4/StatefulPartitionedCall:output:0*
Tin
2*(
_output_shapes
:         А**
config_proto

GPU 

CPU2J 8*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119654*
Tout
2*.
_gradient_op_typePartitionedCall-6119665╜
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0+output_layer_statefulpartitionedcall_args_1+output_layer_statefulpartitionedcall_args_2*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-6119695*R
fMRK
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2р
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0&^Hidden-Layer1/StatefulPartitionedCall&^Hidden-Layer2/StatefulPartitionedCall&^Hidden-Layer3/StatefulPartitionedCall&^Hidden-Layer4/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2N
%Hidden-Layer1/StatefulPartitionedCall%Hidden-Layer1/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall2N
%Hidden-Layer2/StatefulPartitionedCall%Hidden-Layer2/StatefulPartitionedCall2N
%Hidden-Layer3/StatefulPartitionedCall%Hidden-Layer3/StatefulPartitionedCall2N
%Hidden-Layer4/StatefulPartitionedCall%Hidden-Layer4/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall: : :	 :
 :/ +
)
_user_specified_nameflatten_2_input: : : : : : 
╥	
у
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119981

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
TanhTanhBiasAdd:output:0*(
_output_shapes
:         А*
T0В
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*.
_input_shapes
:         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
є
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119509

inputs
identity^
Reshape/shapeConst*
valueB"       *
_output_shapes
:*
dtype0d
ReshapeReshapeinputsReshape/shape:output:0*'
_output_shapes
:         *
T0X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
°
┘
.__inference_sequential_2_layer_call_fn_6119807
flatten_2_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallflatten_2_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10**
config_proto

GPU 

CPU2J 8*
Tout
2*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-6119794*
Tin
2*R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119793В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :	 :
 :/ +
)
_user_specified_nameflatten_2_input: : : : 
╘	
у
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         АВ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ч
░
/__inference_Hidden-Layer1_layer_call_fn_6119988

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6119539**
config_proto

GPU 

CPU2J 8*S
fNRL
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119533*
Tout
2*(
_output_shapes
:         А*
Tin
2Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         А*
T0"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Й
d
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119661

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:         А*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:         А*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
╜
G
+__inference_dropout_2_layer_call_fn_6120077

inputs
identityЮ
PartitionedCallPartitionedCallinputs*
Tout
2*(
_output_shapes
:         А*O
fJRH
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119661*.
_gradient_op_typePartitionedCall-6119673**
config_proto

GPU 

CPU2J 8*
Tin
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
є
b
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119965

inputs
identity^
Reshape/shapeConst*
valueB"       *
dtype0*
_output_shapes
:d
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:         X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*&
_input_shapes
:         :& "
 
_user_specified_nameinputs
│
e
F__inference_dropout_2_layer_call_and_return_conditional_losses_6119654

inputs
identityИQ
dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  А?Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         АМ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0г
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         АХ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*(
_output_shapes
:         А*
T0R
dropout/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  А?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: К
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         Аb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*'
_input_shapes
:         А:& "
 
_user_specified_nameinputs
ш
░
/__inference_Hidden-Layer3_layer_call_fn_6120024

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-6119595*
Tout
2*(
_output_shapes
:         А*
Tin
2*S
fNRL
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6119589**
config_proto

GPU 

CPU2J 8Г
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
╘	
у
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119561

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аб
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
TanhTanhBiasAdd:output:0*(
_output_shapes
:         А*
T0В
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ЇФ
р
#__inference__traced_restore_6120355
file_prefix)
%assignvariableop_hidden_layer1_kernel)
%assignvariableop_1_hidden_layer1_bias+
'assignvariableop_2_hidden_layer2_kernel)
%assignvariableop_3_hidden_layer2_bias+
'assignvariableop_4_hidden_layer3_kernel)
%assignvariableop_5_hidden_layer3_bias+
'assignvariableop_6_hidden_layer4_kernel)
%assignvariableop_7_hidden_layer4_bias,
(assignvariableop_8_output_layer_2_kernel*
&assignvariableop_9_output_layer_2_bias,
(assignvariableop_10_training_2_adam_iter.
*assignvariableop_11_training_2_adam_beta_1.
*assignvariableop_12_training_2_adam_beta_2-
)assignvariableop_13_training_2_adam_decay5
1assignvariableop_14_training_2_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count>
:assignvariableop_17_training_2_adam_hidden_layer1_kernel_m<
8assignvariableop_18_training_2_adam_hidden_layer1_bias_m>
:assignvariableop_19_training_2_adam_hidden_layer2_kernel_m<
8assignvariableop_20_training_2_adam_hidden_layer2_bias_m>
:assignvariableop_21_training_2_adam_hidden_layer3_kernel_m<
8assignvariableop_22_training_2_adam_hidden_layer3_bias_m>
:assignvariableop_23_training_2_adam_hidden_layer4_kernel_m<
8assignvariableop_24_training_2_adam_hidden_layer4_bias_m?
;assignvariableop_25_training_2_adam_output_layer_2_kernel_m=
9assignvariableop_26_training_2_adam_output_layer_2_bias_m>
:assignvariableop_27_training_2_adam_hidden_layer1_kernel_v<
8assignvariableop_28_training_2_adam_hidden_layer1_bias_v>
:assignvariableop_29_training_2_adam_hidden_layer2_kernel_v<
8assignvariableop_30_training_2_adam_hidden_layer2_bias_v>
:assignvariableop_31_training_2_adam_hidden_layer3_kernel_v<
8assignvariableop_32_training_2_adam_hidden_layer3_bias_v>
:assignvariableop_33_training_2_adam_hidden_layer4_kernel_v<
8assignvariableop_34_training_2_adam_hidden_layer4_bias_v?
;assignvariableop_35_training_2_adam_output_layer_2_kernel_v=
9assignvariableop_36_training_2_adam_output_layer_2_bias_v
identity_38ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1т
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*И
value■B√%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0║
RestoreV2/shape_and_slicesConst"/device:CPU:0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:%┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*к
_output_shapesЧ
Ф:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0Б
AssignVariableOpAssignVariableOp%assignvariableop_hidden_layer1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:Е
AssignVariableOp_1AssignVariableOp%assignvariableop_1_hidden_layer1_biasIdentity_1:output:0*
_output_shapes
 *
dtype0N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:З
AssignVariableOp_2AssignVariableOp'assignvariableop_2_hidden_layer2_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0Е
AssignVariableOp_3AssignVariableOp%assignvariableop_3_hidden_layer2_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:З
AssignVariableOp_4AssignVariableOp'assignvariableop_4_hidden_layer3_kernelIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0Е
AssignVariableOp_5AssignVariableOp%assignvariableop_5_hidden_layer3_biasIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:З
AssignVariableOp_6AssignVariableOp'assignvariableop_6_hidden_layer4_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0Е
AssignVariableOp_7AssignVariableOp%assignvariableop_7_hidden_layer4_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:И
AssignVariableOp_8AssignVariableOp(assignvariableop_8_output_layer_2_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0Ж
AssignVariableOp_9AssignVariableOp&assignvariableop_9_output_layer_2_biasIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:К
AssignVariableOp_10AssignVariableOp(assignvariableop_10_training_2_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:М
AssignVariableOp_11AssignVariableOp*assignvariableop_11_training_2_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype0P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0М
AssignVariableOp_12AssignVariableOp*assignvariableop_12_training_2_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype0P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:Л
AssignVariableOp_13AssignVariableOp)assignvariableop_13_training_2_adam_decayIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:У
AssignVariableOp_14AssignVariableOp1assignvariableop_14_training_2_adam_learning_rateIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0{
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:{
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
_output_shapes
:*
T0Ь
AssignVariableOp_17AssignVariableOp:assignvariableop_17_training_2_adam_hidden_layer1_kernel_mIdentity_17:output:0*
_output_shapes
 *
dtype0P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Ъ
AssignVariableOp_18AssignVariableOp8assignvariableop_18_training_2_adam_hidden_layer1_bias_mIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
_output_shapes
:*
T0Ь
AssignVariableOp_19AssignVariableOp:assignvariableop_19_training_2_adam_hidden_layer2_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype0P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Ъ
AssignVariableOp_20AssignVariableOp8assignvariableop_20_training_2_adam_hidden_layer2_bias_mIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0Ь
AssignVariableOp_21AssignVariableOp:assignvariableop_21_training_2_adam_hidden_layer3_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype0P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0Ъ
AssignVariableOp_22AssignVariableOp8assignvariableop_22_training_2_adam_hidden_layer3_bias_mIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:Ь
AssignVariableOp_23AssignVariableOp:assignvariableop_23_training_2_adam_hidden_layer4_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype0P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Ъ
AssignVariableOp_24AssignVariableOp8assignvariableop_24_training_2_adam_hidden_layer4_bias_mIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:Э
AssignVariableOp_25AssignVariableOp;assignvariableop_25_training_2_adam_output_layer_2_kernel_mIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:Ы
AssignVariableOp_26AssignVariableOp9assignvariableop_26_training_2_adam_output_layer_2_bias_mIdentity_26:output:0*
dtype0*
_output_shapes
 P
Identity_27IdentityRestoreV2:tensors:27*
_output_shapes
:*
T0Ь
AssignVariableOp_27AssignVariableOp:assignvariableop_27_training_2_adam_hidden_layer1_kernel_vIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:Ъ
AssignVariableOp_28AssignVariableOp8assignvariableop_28_training_2_adam_hidden_layer1_bias_vIdentity_28:output:0*
_output_shapes
 *
dtype0P
Identity_29IdentityRestoreV2:tensors:29*
_output_shapes
:*
T0Ь
AssignVariableOp_29AssignVariableOp:assignvariableop_29_training_2_adam_hidden_layer2_kernel_vIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
_output_shapes
:*
T0Ъ
AssignVariableOp_30AssignVariableOp8assignvariableop_30_training_2_adam_hidden_layer2_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype0P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0Ь
AssignVariableOp_31AssignVariableOp:assignvariableop_31_training_2_adam_hidden_layer3_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype0P
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:Ъ
AssignVariableOp_32AssignVariableOp8assignvariableop_32_training_2_adam_hidden_layer3_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype0P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0Ь
AssignVariableOp_33AssignVariableOp:assignvariableop_33_training_2_adam_hidden_layer4_kernel_vIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
_output_shapes
:*
T0Ъ
AssignVariableOp_34AssignVariableOp8assignvariableop_34_training_2_adam_hidden_layer4_bias_vIdentity_34:output:0*
dtype0*
_output_shapes
 P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0Э
AssignVariableOp_35AssignVariableOp;assignvariableop_35_training_2_adam_output_layer_2_kernel_vIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:Ы
AssignVariableOp_36AssignVariableOp9assignvariableop_36_training_2_adam_output_layer_2_bias_vIdentity_36:output:0*
dtype0*
_output_shapes
 М
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¤
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: К
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_38Identity_38:output:0*л
_input_shapesЩ
Ц: :::::::::::::::::::::::::::::::::::::2*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_27: : : : :  :! :" :# :$ :% :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
╓	
т
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6119689

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpг
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Д
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*/
_input_shapes
:         А::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
╨E
▐
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119887

inputs0
,hidden_layer1_matmul_readvariableop_resource1
-hidden_layer1_biasadd_readvariableop_resource0
,hidden_layer2_matmul_readvariableop_resource1
-hidden_layer2_biasadd_readvariableop_resource0
,hidden_layer3_matmul_readvariableop_resource1
-hidden_layer3_biasadd_readvariableop_resource0
,hidden_layer4_matmul_readvariableop_resource1
-hidden_layer4_biasadd_readvariableop_resource/
+output_layer_matmul_readvariableop_resource0
,output_layer_biasadd_readvariableop_resource
identityИв$Hidden-Layer1/BiasAdd/ReadVariableOpв#Hidden-Layer1/MatMul/ReadVariableOpв$Hidden-Layer2/BiasAdd/ReadVariableOpв#Hidden-Layer2/MatMul/ReadVariableOpв$Hidden-Layer3/BiasAdd/ReadVariableOpв#Hidden-Layer3/MatMul/ReadVariableOpв$Hidden-Layer4/BiasAdd/ReadVariableOpв#Hidden-Layer4/MatMul/ReadVariableOpв#Output-Layer/BiasAdd/ReadVariableOpв"Output-Layer/MatMul/ReadVariableOph
flatten_2/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"       x
flatten_2/ReshapeReshapeinputs flatten_2/Reshape/shape:output:0*'
_output_shapes
:         *
T0┐
#Hidden-Layer1/MatMul/ReadVariableOpReadVariableOp,hidden_layer1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	А*
dtype0Ъ
Hidden-Layer1/MatMulMatMulflatten_2/Reshape:output:0+Hidden-Layer1/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╜
$Hidden-Layer1/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:Аб
Hidden-Layer1/BiasAddBiasAddHidden-Layer1/MatMul:product:0,Hidden-Layer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer1/TanhTanhHidden-Layer1/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer2/MatMul/ReadVariableOpReadVariableOp,hidden_layer2_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААЦ
Hidden-Layer2/MatMulMatMulHidden-Layer1/Tanh:y:0+Hidden-Layer2/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0╜
$Hidden-Layer2/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer2_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0б
Hidden-Layer2/BiasAddBiasAddHidden-Layer2/MatMul:product:0,Hidden-Layer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer2/TanhTanhHidden-Layer2/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer3/MatMul/ReadVariableOpReadVariableOp,hidden_layer3_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААЦ
Hidden-Layer3/MatMulMatMulHidden-Layer2/Tanh:y:0+Hidden-Layer3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╜
$Hidden-Layer3/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer3_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0б
Hidden-Layer3/BiasAddBiasAddHidden-Layer3/MatMul:product:0,Hidden-Layer3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer3/TanhTanhHidden-Layer3/BiasAdd:output:0*
T0*(
_output_shapes
:         А└
#Hidden-Layer4/MatMul/ReadVariableOpReadVariableOp,hidden_layer4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААЦ
Hidden-Layer4/MatMulMatMulHidden-Layer3/Tanh:y:0+Hidden-Layer4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╜
$Hidden-Layer4/BiasAdd/ReadVariableOpReadVariableOp-hidden_layer4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0б
Hidden-Layer4/BiasAddBiasAddHidden-Layer4/MatMul:product:0,Hidden-Layer4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аm
Hidden-Layer4/TanhTanhHidden-Layer4/BiasAdd:output:0*(
_output_shapes
:         А*
T0[
dropout_2/dropout/rateConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>]
dropout_2/dropout/ShapeShapeHidden-Layer4/Tanh:y:0*
T0*
_output_shapes
:i
$dropout_2/dropout/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    i
$dropout_2/dropout/random_uniform/maxConst*
valueB
 *  А?*
_output_shapes
: *
dtype0б
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*(
_output_shapes
:         А*
dtype0*
T0к
$dropout_2/dropout/random_uniform/subSub-dropout_2/dropout/random_uniform/max:output:0-dropout_2/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ┴
$dropout_2/dropout/random_uniform/mulMul7dropout_2/dropout/random_uniform/RandomUniform:output:0(dropout_2/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         А│
 dropout_2/dropout/random_uniformAdd(dropout_2/dropout/random_uniform/mul:z:0-dropout_2/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         А\
dropout_2/dropout/sub/xConst*
valueB
 *  А?*
_output_shapes
: *
dtype0А
dropout_2/dropout/subSub dropout_2/dropout/sub/x:output:0dropout_2/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_2/dropout/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Ж
dropout_2/dropout/truedivRealDiv$dropout_2/dropout/truediv/x:output:0dropout_2/dropout/sub:z:0*
T0*
_output_shapes
: и
dropout_2/dropout/GreaterEqualGreaterEqual$dropout_2/dropout/random_uniform:z:0dropout_2/dropout/rate:output:0*(
_output_shapes
:         А*
T0Ж
dropout_2/dropout/mulMulHidden-Layer4/Tanh:y:0dropout_2/dropout/truediv:z:0*
T0*(
_output_shapes
:         АД
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АИ
dropout_2/dropout/mul_1Muldropout_2/dropout/mul:z:0dropout_2/dropout/Cast:y:0*
T0*(
_output_shapes
:         А╜
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	АШ
Output-Layer/MatMulMatMuldropout_2/dropout/mul_1:z:0*Output-Layer/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0║
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Э
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0p
Output-Layer/SigmoidSigmoidOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:         ▀
IdentityIdentityOutput-Layer/Sigmoid:y:0%^Hidden-Layer1/BiasAdd/ReadVariableOp$^Hidden-Layer1/MatMul/ReadVariableOp%^Hidden-Layer2/BiasAdd/ReadVariableOp$^Hidden-Layer2/MatMul/ReadVariableOp%^Hidden-Layer3/BiasAdd/ReadVariableOp$^Hidden-Layer3/MatMul/ReadVariableOp%^Hidden-Layer4/BiasAdd/ReadVariableOp$^Hidden-Layer4/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::2L
$Hidden-Layer2/BiasAdd/ReadVariableOp$Hidden-Layer2/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp2J
#Hidden-Layer1/MatMul/ReadVariableOp#Hidden-Layer1/MatMul/ReadVariableOp2L
$Hidden-Layer1/BiasAdd/ReadVariableOp$Hidden-Layer1/BiasAdd/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2J
#Hidden-Layer3/MatMul/ReadVariableOp#Hidden-Layer3/MatMul/ReadVariableOp2L
$Hidden-Layer4/BiasAdd/ReadVariableOp$Hidden-Layer4/BiasAdd/ReadVariableOp2J
#Hidden-Layer2/MatMul/ReadVariableOp#Hidden-Layer2/MatMul/ReadVariableOp2L
$Hidden-Layer3/BiasAdd/ReadVariableOp$Hidden-Layer3/BiasAdd/ReadVariableOp2J
#Hidden-Layer4/MatMul/ReadVariableOp#Hidden-Layer4/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
╘	
у
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6119617

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpд
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
ААj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0б
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         А*
T0Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:         АВ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*/
_input_shapes
:         А::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
▌
╨
.__inference_sequential_2_layer_call_fn_6119944

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*'
_output_shapes
:         *R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119754*
Tout
2*.
_gradient_op_typePartitionedCall-6119755*
Tin
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 
▌
╨
.__inference_sequential_2_layer_call_fn_6119959

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10
identityИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10*'
_output_shapes
:         *
Tin
2*R
fMRK
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119793*.
_gradient_op_typePartitionedCall-6119794*
Tout
2**
config_proto

GPU 

CPU2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*N
_input_shapes=
;:         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*┐
serving_defaultл
K
flatten_2_input8
!serving_default_flatten_2_input:0         @
Output-Layer0
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:■щ
°0
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
		optimizer

trainable_variables
regularization_losses
	variables
	keras_api

signatures
+В&call_and_return_all_conditional_losses
Г__call__
Д_default_save_signature"└-
_tf_keras_sequentialб-{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_2", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "Hidden-Layer4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
│
trainable_variables
regularization_losses
	variables
	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"в
_tf_keras_layerИ{"class_name": "InputLayer", "name": "flatten_2_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"batch_input_shape": [null, 8], "dtype": "float32", "sparse": false, "name": "flatten_2_input"}}
╫
trainable_variables
regularization_losses
	variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"╞
_tf_keras_layerм{"class_name": "Flatten", "name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 8], "config": {"name": "flatten_2", "trainable": true, "batch_input_shape": [null, 8], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"╪
_tf_keras_layer╛{"class_name": "Dense", "name": "Hidden-Layer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Hidden-Layer1", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}}
Б

kernel
bias
trainable_variables
 regularization_losses
!	variables
"	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"┌
_tf_keras_layer└{"class_name": "Dense", "name": "Hidden-Layer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Hidden-Layer2", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Б

#kernel
$bias
%trainable_variables
&regularization_losses
'	variables
(	keras_api
+Н&call_and_return_all_conditional_losses
О__call__"┌
_tf_keras_layer└{"class_name": "Dense", "name": "Hidden-Layer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Hidden-Layer3", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Б

)kernel
*bias
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+П&call_and_return_all_conditional_losses
Р__call__"┌
_tf_keras_layer└{"class_name": "Dense", "name": "Hidden-Layer4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Hidden-Layer4", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
▒
/trainable_variables
0regularization_losses
1	variables
2	keras_api
+С&call_and_return_all_conditional_losses
Т__call__"а
_tf_keras_layerЖ{"class_name": "Dropout", "name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
А

3kernel
4bias
5trainable_variables
6regularization_losses
7	variables
8	keras_api
+У&call_and_return_all_conditional_losses
Ф__call__"┘
_tf_keras_layer┐{"class_name": "Dense", "name": "Output-Layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Output-Layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Й
9iter

:beta_1

;beta_2
	<decay
=learning_ratemnmompmq#mr$ms)mt*mu3mv4mwvxvyvzv{#v|$v})v~*v3vА4vБ"
	optimizer
f
0
1
2
3
#4
$5
)6
*7
38
49"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
#4
$5
)6
*7
38
49"
trackable_list_wrapper
╗
>metrics

trainable_variables

?layers
regularization_losses
@non_trainable_variables
	variables
Alayer_regularization_losses
Г__call__
Д_default_save_signature
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
-
Хserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Bmetrics
trainable_variables

Clayers
Dnon_trainable_variables
regularization_losses
	variables
Elayer_regularization_losses
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Fmetrics
trainable_variables

Glayers
Hnon_trainable_variables
regularization_losses
	variables
Ilayer_regularization_losses
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
':%	А2Hidden-Layer1/kernel
!:А2Hidden-Layer1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Э
Jmetrics
trainable_variables

Klayers
Lnon_trainable_variables
regularization_losses
	variables
Mlayer_regularization_losses
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
(:&
АА2Hidden-Layer2/kernel
!:А2Hidden-Layer2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Э
Nmetrics
trainable_variables

Olayers
Pnon_trainable_variables
 regularization_losses
!	variables
Qlayer_regularization_losses
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
(:&
АА2Hidden-Layer3/kernel
!:А2Hidden-Layer3/bias
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
Э
Rmetrics
%trainable_variables

Slayers
Tnon_trainable_variables
&regularization_losses
'	variables
Ulayer_regularization_losses
О__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
(:&
АА2Hidden-Layer4/kernel
!:А2Hidden-Layer4/bias
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
Э
Vmetrics
+trainable_variables

Wlayers
Xnon_trainable_variables
,regularization_losses
-	variables
Ylayer_regularization_losses
Р__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Э
Zmetrics
/trainable_variables

[layers
\non_trainable_variables
0regularization_losses
1	variables
]layer_regularization_losses
Т__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
(:&	А2Output-Layer_2/kernel
!:2Output-Layer_2/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
Э
^metrics
5trainable_variables

_layers
`non_trainable_variables
6regularization_losses
7	variables
alayer_regularization_losses
Ф__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_2/Adam/iter
 : (2training_2/Adam/beta_1
 : (2training_2/Adam/beta_2
: (2training_2/Adam/decay
':% (2training_2/Adam/learning_rate
'
b0"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
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
Ь
	ctotal
	dcount
e
_fn_kwargs
ftrainable_variables
gregularization_losses
h	variables
i	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"х
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
Э
jmetrics
ftrainable_variables

klayers
lnon_trainable_variables
gregularization_losses
h	variables
mlayer_regularization_losses
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
7:5	А2&training_2/Adam/Hidden-Layer1/kernel/m
1:/А2$training_2/Adam/Hidden-Layer1/bias/m
8:6
АА2&training_2/Adam/Hidden-Layer2/kernel/m
1:/А2$training_2/Adam/Hidden-Layer2/bias/m
8:6
АА2&training_2/Adam/Hidden-Layer3/kernel/m
1:/А2$training_2/Adam/Hidden-Layer3/bias/m
8:6
АА2&training_2/Adam/Hidden-Layer4/kernel/m
1:/А2$training_2/Adam/Hidden-Layer4/bias/m
8:6	А2'training_2/Adam/Output-Layer_2/kernel/m
1:/2%training_2/Adam/Output-Layer_2/bias/m
7:5	А2&training_2/Adam/Hidden-Layer1/kernel/v
1:/А2$training_2/Adam/Hidden-Layer1/bias/v
8:6
АА2&training_2/Adam/Hidden-Layer2/kernel/v
1:/А2$training_2/Adam/Hidden-Layer2/bias/v
8:6
АА2&training_2/Adam/Hidden-Layer3/kernel/v
1:/А2$training_2/Adam/Hidden-Layer3/bias/v
8:6
АА2&training_2/Adam/Hidden-Layer4/kernel/v
1:/А2$training_2/Adam/Hidden-Layer4/bias/v
8:6	А2'training_2/Adam/Output-Layer_2/kernel/v
1:/2%training_2/Adam/Output-Layer_2/bias/v
Є2я
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119707
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119929
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119887
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119730└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ж2Г
.__inference_sequential_2_layer_call_fn_6119768
.__inference_sequential_2_layer_call_fn_6119944
.__inference_sequential_2_layer_call_fn_6119959
.__inference_sequential_2_layer_call_fn_6119807└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ш2х
"__inference__wrapped_model_6119497╛
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *.в+
)К&
flatten_2_input         
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
Ё2э
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119965в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╒2╥
+__inference_flatten_2_layer_call_fn_6119970в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119981в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_Hidden-Layer1_layer_call_fn_6119988в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119999в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_Hidden-Layer2_layer_call_fn_6120006в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6120017в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_Hidden-Layer3_layer_call_fn_6120024в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6120035в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
/__inference_Hidden-Layer4_layer_call_fn_6120042в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╩2╟
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120062
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120067┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ф2С
+__inference_dropout_2_layer_call_fn_6120077
+__inference_dropout_2_layer_call_fn_6120072┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
є2Ё
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6120088в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╪2╒
.__inference_Output-Layer_layer_call_fn_6120095в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
<B:
%__inference_signature_wrapper_6119828flatten_2_input
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 ┬
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119707u
#$)*34@в=
6в3
)К&
flatten_2_input         
p

 
к "%в"
К
0         
Ъ А
+__inference_dropout_2_layer_call_fn_6120072Q4в1
*в'
!К
inputs         А
p
к "К         АС
.__inference_sequential_2_layer_call_fn_6119959_
#$)*347в4
-в*
 К
inputs         
p 

 
к "К         л
J__inference_Hidden-Layer1_layer_call_and_return_conditional_losses_6119981]/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ Д
/__inference_Hidden-Layer4_layer_call_fn_6120042Q)*0в-
&в#
!К
inputs         А
к "К         АЪ
.__inference_sequential_2_layer_call_fn_6119807h
#$)*34@в=
6в3
)К&
flatten_2_input         
p 

 
к "К         Ъ
.__inference_sequential_2_layer_call_fn_6119768h
#$)*34@в=
6в3
)К&
flatten_2_input         
p

 
к "К         к
"__inference__wrapped_model_6119497Г
#$)*348в5
.в+
)К&
flatten_2_input         
к ";к8
6
Output-Layer&К#
Output-Layer         └
%__inference_signature_wrapper_6119828Ц
#$)*34KвH
в 
Aк>
<
flatten_2_input)К&
flatten_2_input         ";к8
6
Output-Layer&К#
Output-Layer         С
.__inference_sequential_2_layer_call_fn_6119944_
#$)*347в4
-в*
 К
inputs         
p

 
к "К         м
J__inference_Hidden-Layer3_layer_call_and_return_conditional_losses_6120017^#$0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ в
F__inference_flatten_2_layer_call_and_return_conditional_losses_6119965X/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ м
J__inference_Hidden-Layer4_layer_call_and_return_conditional_losses_6120035^)*0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ ╣
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119929l
#$)*347в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ к
I__inference_Output-Layer_layer_call_and_return_conditional_losses_6120088]340в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ и
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120062^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Г
/__inference_Hidden-Layer1_layer_call_fn_6119988P/в,
%в"
 К
inputs         
к "К         Ам
J__inference_Hidden-Layer2_layer_call_and_return_conditional_losses_6119999^0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ и
F__inference_dropout_2_layer_call_and_return_conditional_losses_6120067^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ В
.__inference_Output-Layer_layer_call_fn_6120095P340в-
&в#
!К
inputs         А
к "К         ╣
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119887l
#$)*347в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ Д
/__inference_Hidden-Layer2_layer_call_fn_6120006Q0в-
&в#
!К
inputs         А
к "К         АА
+__inference_dropout_2_layer_call_fn_6120077Q4в1
*в'
!К
inputs         А
p 
к "К         Аz
+__inference_flatten_2_layer_call_fn_6119970K/в,
%в"
 К
inputs         
к "К         ┬
I__inference_sequential_2_layer_call_and_return_conditional_losses_6119730u
#$)*34@в=
6в3
)К&
flatten_2_input         
p 

 
к "%в"
К
0         
Ъ Д
/__inference_Hidden-Layer3_layer_call_fn_6120024Q#$0в-
&в#
!К
inputs         А
к "К         А