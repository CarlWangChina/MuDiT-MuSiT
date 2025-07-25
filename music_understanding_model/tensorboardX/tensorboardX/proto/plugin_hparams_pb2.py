import sys
b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from tensorboardX.proto import api_pb2 as tensorboardX_dot_proto_dot_api__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2

DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorboardX/proto/plugin_hparams.proto',
  package='tensorboardX.hparam',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\'tensorboardX/proto/plugin_hparams.proto\x12\x13tensorboardX.hparam\x1a\x1ctensorboardX/proto/api.proto\x1a\x1cgoogle/protobuf/struct.proto\"\xe9\x01\n\x11HParamsPluginData\x12\x0f\n\x07version\x18\x01 \x01(\x05\x12\x35\n\nexperiment\x18\x02 \x01(\x0b\x32\x1f.tensorboardX.hparam.ExperimentH\x00\x12\x43\n\x12session_start_info\x18\x03 \x01(\x0b\x32%.tensorboardX.hparam.SessionStartInfoH\x00\x12?\n\x10session_end_info\x18\x04 \x01(\x0b\x32 .tensorboardX.hparam.SessionEndInfoH\x00\x42\x05\n\x03\x64\x61ta\"\xe3\x02\n\x11SessionStartInfo\x12R\n\x07hparams\x18\x01 \x03(\x0b\x32=.tensorboardX.hparam.SessionStartInfo.HparamsEntry\x12\x11\n\tmodel_uri\x18\x02 \x01(\t\x12\x13\n\x0bmonitor_url\x18\x03 \x01(\t\x12\x12\n\ngroup_name\x18\x04 \x01(\t\x12\x17\n\x0fstart_time_secs\x18\x05 \x01(\x01\x1a\x44\n\x0cHparamsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b\x32\x14.google.protobuf.Value:\x02\x38\x01\"\\\n\x0eSessionEndInfo\x12\x37\n\x06status\x18\x01 \x01(\x0e\x32\'.tensorboardX.proto.api_pb2.Status\x12\x17\n\x0f\x65nd_time_secs\x18\x02 \x01(\x01\x42\x05\n\x03\x64\x61ta')
)


_HPARAMSPLUGINDATA = _descriptor.Descriptor(
  name='HParamsPluginData',
  full_name='tensorboardX.hparam.HParamsPluginData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='version', full_name='tensorboardX.hparam.HParamsPluginData.version', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='experiment', full_name='tensorboardX.hparam.HParamsPluginData.experiment', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='session_start_info', full_name='tensorboardX.hparam.HParamsPluginData.session_start_info', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='session_end_info', full_name='tensorboardX.hparam.HParamsPluginData.session_end_info', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='tensorboardX.hparam.HParamsPluginData.data',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=125,
  serialized_end=358,
)


_SESSIONSTARTINFO_HPARAMSENTRY = _descriptor.Descriptor(
  name='HparamsEntry',
  full_name='tensorboardX.hparam.SessionStartInfo.HparamsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='tensorboardX.hparam.SessionStartInfo.HparamsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='value', full_name='tensorboardX.hparam.SessionStartInfo.HparamsEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=_b('8\001'),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=535,
  serialized_end=605,
)

_SESSIONSTARTINFO = _descriptor.Descriptor(
  name='SessionStartInfo',
  full_name='tensorboardX.hparam.SessionStartInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='hparams', full_name='tensorboardX.hparam.SessionStartInfo.hparams', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_uri', full_name='tensorboardX.hparam.SessionStartInfo.model_uri', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='monitor_url', full_name='tensorboardX.hparam.SessionStartInfo.monitor_url', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='group_name', full_name='tensorboardX.hparam.SessionStartInfo.group_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='start_time_secs', full_name='tensorboardX.hparam.SessionStartInfo.start_time_secs', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_SESSIONSTARTINFO_HPARAMSENTRY, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=361,
  serialized_end=605,
)


_SESSIONENDINFO = _descriptor.Descriptor(
  name='SessionEndInfo',
  full_name='tensorboardX.hparam.SessionEndInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='tensorboardX.hparam.SessionEndInfo.status', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='end_time_secs', full_name='tensorboardX.hparam.SessionEndInfo.end_time_secs', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=607,
  serialized_end=691,
)

_HPARAMSPLUGINDATA.fields_by_name['experiment'].message_type = tensorboardX_dot_proto_dot_api__pb2._EXPERIMENT
_HPARAMSPLUGINDATA.fields_by_name['session_start_info'].message_type = _SESSIONSTARTINFO
_HPARAMSPLUGINDATA.fields_by_name['session_end_info'].message_type = _SESSIONENDINFO
_HPARAMSPLUGINDATA.oneofs_by_name['data'].fields.append(
  _HPARAMSPLUGINDATA.fields_by_name['experiment'])
_HPARAMSPLUGINDATA.fields_by_name['experiment'].containing_oneof = _HPARAMSPLUGINDATA.oneofs_by_name['data']
_HPARAMSPLUGINDATA.oneofs_by_name['data'].fields.append(
  _HPARAMSPLUGINDATA.fields_by_name['session_start_info'])
_HPARAMSPLUGINDATA.fields_by_name['session_start_info'].containing_oneof = _HPARAMSPLUGINDATA.oneofs_by_name['data']
_HPARAMSPLUGINDATA.oneofs_by_name['data'].fields.append(
  _HPARAMSPLUGINDATA.fields_by_name['session_end_info'])
_HPARAMSPLUGINDATA.fields_by_name['session_end_info'].containing_oneof = _HPARAMSPLUGINDATA.oneofs_by_name['data']
_SESSIONSTARTINFO_HPARAMSENTRY.fields_by_name['value'].message_type = google_dot_protobuf_dot_struct__pb2._VALUE
_SESSIONSTARTINFO_HPARAMSENTRY.containing_type = _SESSIONSTARTINFO
_SESSIONSTARTINFO.fields_by_name['hparams'].message_type = _SESSIONSTARTINFO_HPARAMSENTRY
_SESSIONENDINFO.fields_by_name['status'].enum_type = tensorboardX_dot_proto_dot_api__pb2._STATUS
DESCRIPTOR.message_types_by_name['HParamsPluginData'] = _HPARAMSPLUGINDATA
DESCRIPTOR.message_types_by_name['SessionStartInfo'] = _SESSIONSTARTINFO
DESCRIPTOR.message_types_by_name['SessionEndInfo'] = _SESSIONENDINFO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HParamsPluginData = _reflection.GeneratedProtocolMessageType('HParamsPluginData', (_message.Message,), dict(
  DESCRIPTOR = _HPARAMSPLUGINDATA,
  __module__ = 'tensorboardX.proto.plugin_hparams_pb2'
  ))
_sym_db.RegisterMessage(HParamsPluginData)

SessionStartInfo = _reflection.GeneratedProtocolMessageType('SessionStartInfo', (_message.Message,), dict(
  HparamsEntry = _reflection.GeneratedProtocolMessageType('HparamsEntry', (_message.Message,), dict(
    DESCRIPTOR = _SESSIONSTARTINFO_HPARAMSENTRY,
    __module__ = 'tensorboardX.proto.plugin_hparams_pb2'
    )),
  DESCRIPTOR = _SESSIONSTARTINFO,
  __module__ = 'tensorboardX.proto.plugin_hparams_pb2'
  ))
_sym_db.RegisterMessage(SessionStartInfo)
_sym_db.RegisterMessage(SessionStartInfo.HparamsEntry)

SessionEndInfo = _reflection.GeneratedProtocolMessageType('SessionEndInfo', (_message.Message,), dict(
  DESCRIPTOR = _SESSIONENDINFO,
  __module__ = 'tensorboardX.proto.plugin_hparams_pb2'
  ))
_sym_db.RegisterMessage(SessionEndInfo)

_SESSIONSTARTINFO_HPARAMSENTRY._options = None