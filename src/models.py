import datetime
from typing import TypedDict

from peewee import *

import assets

db_event_base = SqliteDatabase(assets.db_uri_database())


class BaseModel(Model):
    class Meta:
        database = db_event_base


class Event(BaseModel):
    id = BigAutoField(primary_key=True)
    content = TextField(help_text='事件内容')
    record_at = DateTimeField(default=datetime.datetime.now, help_text='content的一部分,表示事件发生时刻')
    # parent = ForeignKeyField('self', null=True, lazy_load=True,
    #                          help_text='对于关联事件, 可指定parent')
    create_at = DateTimeField(default=datetime.datetime.now, help_text='event的创建时间')
    update_at = DateTimeField(default=datetime.datetime.now, help_text='event.content的更新时间')
    deleted = BooleanField(default=False)


class Tag(BaseModel):
    id = AutoField(primary_key=True)
    name = CharField()
    parent = ForeignKeyField('self', backref='children', null=True, lazy_load=True,
                             help_text='某些Tag之间存在包含关系')
    desc = CharField(default='', help_text='标签描述')


class EventTag(BaseModel):
    """Tag与Event多对多关系
    不要使用ManyToMany
    tags: 指 MindEvent实例将有虚拟字段 类型为 List[EventTag],
        EventTag 实例 binding的字段 tag, 类型为 Tag
        tags = [binding.tag.name for binding in e.event_bindings]
    """
    event = ForeignKeyField(Event, backref='tags')
    tag = ForeignKeyField(Tag, backref='events')


class Meta(BaseModel):
    """Event的元数据属性记录
    一条记录代表Event的Meta的其中一个属性, 又因为属性之间可能由歧义,因此meta与特定tag关联
    event->tag->meta
    """
    id = BigAutoField(primary_key=True)
    event_tag = ForeignKeyField(EventTag, backref='metas',
                                help_text='meta属性是建立在 event-tag之上的')
    key = CharField()
    value = TextField()


# ===
class TodoDTO(TypedDict):
    """每一条属性都将被作为一条meta记录存储
    待办的完成时间: event.record_at
    待办的截止时间: meta.due_at, 是todo的属性
    """
    # priority: int = 0  # 优先级  00普通 01重要 10紧急 11重要且紧急
    # status: bool = False
    due_at: datetime.datetime  # 截止
    # remind_at: datetime.datetime = None # 提醒
