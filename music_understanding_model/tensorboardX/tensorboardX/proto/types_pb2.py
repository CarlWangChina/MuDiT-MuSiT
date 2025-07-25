import sys
_b = sys.version_info[0] < 3 and (lambda x: x) or (lambda x: x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor.FileDescriptor(
    name='tensorboardX/proto/types.proto',
    package='tensorboardX',
    syntax='proto3',
    serialized_options=_b('\n\030org.tensorflow.frameworkB\013TypesProtosP\001\370\001\001'),
    serialized_pb=_b('\n\x1etensorboardX/proto/types.proto\x12\x0ctensorboardX*\xc2\x05\n\x08\x44\x61taType\x12\x0e\n\nDT_INVALID\x10\x00\x12\x0c\n\x08\x44T_FLOAT\x10\x01\x12\r\n\tDT_DOUBLE\x10\x02\x12\x0c\n\x08\x44T_INT32\x10\x03\x12\x0c\n\x08\x44T_UINT8\x10\x04\x12\x0c\n\x08\x44T_INT16\x10\x05\x12\x0b\n\x07\x44T_INT8\x10\x06\x12\r\n\tDT_STRING\x10\x07\x12\x10\n\x0c\x44T_COMPLEX64\x10\x08\x12\x0c\n\x08\x44T_INT64\x10\t\x12\x0b\n\x07\x44T_BOOL\x10\n\x12\x0c\n\x08\x44T_QINT8\x10\x0b\x12\r\n\tDT_QUINT8\x10\x0c\x12\r\n\tDT_QINT32\x10\r\x12\x0f\n\x0b\x44T_BFLOAT16\x10\x0e\x12\r\n\tDT_QINT16\x10\x0f\x12\x0e\n\nDT_QUINT16\x10\x10\x12\r\n\tDT_UINT16\x10\x11\x12\x11\n\rDT_COMPLEX128\x10\x12\x12\x0b\n\x07\x44T_HALF\x10\x13\x12\x0f\n\x0b\x44T_RESOURCE\x10\x14\x12\x10\n\x0c\x44T_FLOAT_REF\x10\x65\x12\x11\n\rDT_DOUBLE_REF\x10\x66\x12\x10\n\x0c\x44T_INT32_REF\x10g\x12\x10\n\x0c\x44T_UINT8_REF\x10h\x12\x10\n\x0c\x44T_INT16_REF\x10i\x12\x0f\n\x0b\x44T_INT8_REF\x10j\x12\x11\n\rDT_STRING_REF\x10k\x12\x14\n\x10\x44T_COMPLEX64_REF\x10l\x12\x10\n\x0c\x44T_INT64_REF\x10m\x12\x0f\n\x0b\x44T_BOOL_REF\x10n\x12\x10\n\x0c\x44T_QINT8_REF\x10o\x12\x11\n\rDT_QUINT8_REF\x10p\x12\x11\n\rDT_QINT32_REF\x10q\x12\x13\n\x0f\x44T_BFLOAT16_REF\x10r\x12\x11\n\rDT_QINT16_REF\x10s\x12\x12\n\x0e\x44T_QUINT16_REF\x10t\x12\x11\n\rDT_UINT16_REF\x10u\x12\x15\n\x11\x44T_COMPLEX128_REF\x10v\x12\x0f\n\x0b\x44T_HALF_REF\x10w\x12\x13\n\x0f\x44T_RESOURCE_REF\x10xB,\n\x18org.tensorflow.frameworkB\x0bTypesProtosP\x01\xf8\x01\x01\x62\x06proto3'))
_DATATYPE = _descriptor.EnumDescriptor(
    name='DataType',
    full_name='tensorboardX.DataType',
    filename=None,
    file=DESCRIPTOR,
    values=[
        _descriptor.EnumValueDescriptor(
            name='DT_INVALID', index=0, number=0,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_FLOAT', index=1, number=1,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_DOUBLE', index=2, number=2,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT32', index=3, number=3,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_UINT8', index=4, number=4,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT16', index=5, number=5,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT8', index=6, number=6,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_STRING', index=7, number=7,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_COMPLEX64', index=8, number=8,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT64', index=9, number=9,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_BOOL', index=10, number=10,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT8', index=11, number=11,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QUINT8', index=12, number=12,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT32', index=13, number=13,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_BFLOAT16', index=14, number=14,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT16', index=15, number=15,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QUINT16', index=16, number=16,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_UINT16', index=17, number=17,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_COMPLEX128', index=18, number=18,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_HALF', index=19, number=19,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_RESOURCE', index=20, number=20,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_FLOAT_REF', index=21, number=101,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_DOUBLE_REF', index=22, number=102,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT32_REF', index=23, number=103,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_UINT8_REF', index=24, number=104,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT16_REF', index=25, number=105,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT8_REF', index=26, number=106,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_STRING_REF', index=27, number=107,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_COMPLEX64_REF', index=28, number=108,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_INT64_REF', index=29, number=109,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_BOOL_REF', index=30, number=110,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT8_REF', index=31, number=111,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QUINT8_REF', index=32, number=112,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT32_REF', index=33, number=113,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_BFLOAT16_REF', index=34, number=114,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QINT16_REF', index=35, number=115,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_QUINT16_REF', index=36, number=116,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_UINT16_REF', index=37, number=117,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_COMPLEX128_REF', index=38, number=118,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_HALF_REF', index=39, number=119,
            serialized_options=None,
            type=None),
        _descriptor.EnumValueDescriptor(
            name='DT_RESOURCE_REF', index=40, number=120,
            serialized_options=None,
            type=None),
    ],
    containing_type=None,
    serialized_options=None,
    serialized_start=49,
    serialized_end=755,)
_sym_db.RegisterEnumDescriptor(_DATATYPE)
DataType = enum_type_wrapper.EnumTypeWrapper(_DATATYPE)
DT_INVALID = 0
DT_FLOAT = 1
DT_DOUBLE = 2
DT_INT32 = 3
DT_UINT8 = 4
DT_INT16 = 5
DT_INT8 = 6
DT_STRING = 7
DT_COMPLEX64 = 8
DT_INT64 = 9
DT_BOOL = 10
DT_QINT8 = 11
DT_QUINT8 = 12
DT_QINT32 = 13
DT_BFLOAT16 = 14
DT_QINT16 = 15
DT_QUINT16 = 16
DT_UINT16 = 17
DT_COMPLEX128 = 18
DT_HALF = 19
DT_RESOURCE = 20
DT_FLOAT_REF = 101
DT_DOUBLE_REF = 102
DT_INT32_REF = 103
DT_UINT8_REF = 104
DT_INT16_REF = 105
DT_INT8_REF = 106
DT_STRING_REF = 107
DT_COMPLEX64_REF = 108
DT_INT64_REF = 109
DT_BOOL_REF = 110
DT_QINT8_REF = 111
DT_QUINT8_REF = 112
DT_QINT32_REF = 113
DT_BFLOAT16_REF = 114
DT_QINT16_REF = 115
DT_QUINT16_REF = 116
DT_UINT16_REF = 117
DT_COMPLEX128_REF = 118
DT_HALF_REF = 119
DT_RESOURCE_REF = 120
DESCRIPTOR.enum_types_by_name['DataType'] = _DATATYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)
DESCRIPTOR._options = None