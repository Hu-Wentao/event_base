from datetime import datetime
from typing import Callable, Iterable, Any

import pandas as pd
import streamlit as st
from loguru import logger
from peewee import JOIN, fn

import assets
from src.const import CONST_TAG_ID_TODO, CONST_TAG_ID_SYSTEM
from src.models import Tag, EventTag, Event, TodoDTO, Meta


# === 后端逻辑,函数,ORM ===
def save_event(event: Event, tags: Iterable[Tag]) -> list[EventTag]:
    logger.info(f"{event.content[:10]},{event.record_at}")
    event.save()
    event_tags = [EventTag.create(event=event, tag=t) for t in tags]
    return event_tags


def save_tag(tag: Tag):
    logger.info(f"{tag.name}")
    tag.save()
    now = datetime.now()
    save_event(
        event=Event(
            content=f'CREATE Tag: {tag.name}#{tag.id}',
            record_at=now,
            create_at=now,
            update_at=now, ),
        tags=[Tag.get_by_id(CONST_TAG_ID_SYSTEM)],
    )


def save_todo(event: Event, todo: TodoDTO):
    """创建Todo tag的event"""
    logger.info(f"{todo}")
    todo_tag = Tag.get_by_id(CONST_TAG_ID_TODO)
    binding: EventTag = save_event(event, tags=[todo_tag])[0]  # 只传入1个tag,因此返回1个binding
    binding_id = binding.get_id()
    for kv in todo.items():
        Meta.create(
            event_tag=binding_id,
            key=str(kv[0]),
            value=str(kv[1]),
        )
    pass


def update_event(event_id: int, edits: dict[str, Any]):
    """更新数据表"""
    logger.info(f"{event_id}, {edits}")
    if 'tags' in edits.keys():
        # fixme raise "暂不支持编辑tags"
        raise "暂不支持编辑tags"
    for k, v in edits.items():
        if isinstance(v, datetime):
            edits[k] = v.isoformat()
    rst = (
        Event
        .update({
            **edits,
            'update_at': datetime.now(), })
        .where(Event.id == event_id)
    ).execute()
    logger.debug(f'{rst}')


def update_todo_event(event_id: int, edits: dict[str, Any], ):
    """更新数据表"""
    logger.info(f"{event_id}, {edits}")
    if 'tags' in edits.keys():
        raise "Todo不支持编辑tags!"
    rst = (
        Event
        .update({
            **edits,
            'update_at': datetime.now(),
        })
        .where(Event.id == event_id)
    ).execute()
    logger.debug(f'{rst}')


def query_events(page: int = 1, by: int = 20, show_tag_system=False) -> pd.DataFrame:
    logger.info(f"page:{page},by:{by},system:{show_tag_system}")
    query = (
        Event
        .select(Event, fn.GROUP_CONCAT(Tag.name).alias('tags'))
        .where(Event.deleted == False)
        .join(EventTag, JOIN.LEFT_OUTER)  # 使用左外连接
        .join(Tag, JOIN.LEFT_OUTER)  # 使用左外连接
        .group_by(Event.id)
    )
    if not show_tag_system:
        query = query.where((Tag.id != CONST_TAG_ID_SYSTEM) | (Tag.name.is_null()))  # 排除system标签
    query = (
        query
        .order_by(Event.record_at.desc())
        .paginate(page, by))

    # 此处 tags为新增的alias
    df = pd.DataFrame([{**e.__data__, 'tags': e.tags} for e in query])
    return df


def query_todo_events(page: int = 1, by: int = 20) -> pd.DataFrame:
    logger.info(f"page:{page},by:{by},")
    query = (
        Event
        .select(Event, fn.GROUP_CONCAT(Tag.name).alias('tags'))
        .where((Event.deleted == False))
        .join(EventTag, JOIN.LEFT_OUTER)  # 使用左外连接
        .join(Tag, JOIN.LEFT_OUTER)  # 使用左外连接
        .group_by(Event.id)
        .where(Tag.id == CONST_TAG_ID_TODO)  # 查询带有todo标签的event
    )
    # if not show_tag_system:
    #     query = query.where((Tag.id != CONST_TAG_ID_SYSTEM) | (Tag.name.is_null()))  # 排除system标签
    query = (
        query
        .order_by(Event.record_at.desc())
        .paginate(page, by))

    # 此处 tags为新增的alias
    # todo 还需要新增 todo.due_at
    df = pd.DataFrame([{**e.__data__, 'tags': e.tags} for e in query])
    return df


def query_tags(page: int = 1, by: int = 30) -> pd.DataFrame:
    logger.info(f"page:{page},by:{by}")
    tags = (
        Tag
        .select()
        .paginate(page, by)
    )
    df = pd.DataFrame([{**e.__data__} for e in tags])
    return df


# === 前端逻辑/状态管理 ===
def s_event_df(update: pd.DataFrame = None) -> pd.DataFrame:
    """状态
    :param update: 更新df, 来自数据操作/文件变更
    :return: df缓存
    """
    if 'event_df' not in st.session_state:  # 初始化session
        st.session_state['event_df'] = None
    if update is not None:
        st.session_state['event_df'] = update  # 这行代码导致UI刷新,editor高度变化
    if st.session_state['event_df'] is None:
        reload_event_grid()  # 刚从磁盘读取的原始数据,无需save
    return st.session_state['event_df']


def s_todo_df(update: pd.DataFrame = None) -> pd.DataFrame:
    """状态 todo_df 表
    :param update: 更新df, 来自数据操作/文件变更
    :return: df缓存
    """
    k = 'todo_df'
    if k not in st.session_state:  # 初始化session
        st.session_state[k] = None
    if update is not None:
        st.session_state[k] = update  # 这行代码导致UI刷新,editor高度变化
        # 此处对df进行二次处理

    if st.session_state[k] is None:
        reload_todo_grid()
    return st.session_state[k]


def reload_event_grid():
    """重新获取表格数据,并刷新UI
    当表格查询参数变化,新建/删除event时调用
    """
    s_event_df(update=fetch_event_df())  # 刷新 UI


def reload_todo_grid():
    """包装刷新表格的函数
    可以在此 节流防抖
    """
    s_todo_df(update=fetch_todo_df())
    pass


def on_change_event_df():
    """state自带update用于更新状态, 本函数只用于接收editor的change,转换为state(update)"""
    change = st.session_state['edited_event_df']
    df = s_event_df()
    if edited := change['edited_rows']:  # 编辑列属性 # {2: {'col_name': 'foo'}},
        for row_id, edits in edited.items():
            event_id = df.at[row_id, 'id']
            if 'tags' in edits.keys():
                st.toast("暂不支持编辑tags", icon='⚠️')
            update_event(event_id, edits)
        pass
    if added := change['added_rows']:  # 新增列 [{'col_name':'bar'}]
        if added == [{}]:
            return  # 添加空行不保存直接返回不刷新 (避免打断输入)
        st.toast("暂不支持add记录", icon='⚠️')
        pass
    if deleted := change['deleted_rows']:  # 删除列 [1,2,3]
        for row_id in deleted:
            event_id = df.at[row_id, 'id']
            update_event(event_id, {'deleted': True})
        pass
    # 这里传入新的df, 将导致ui刷新
    reload_event_grid()  # 刷新 df


def on_change_todo_df():
    """state自带update用于更新状态, 本函数只用于接收editor的change,转换为state(update)"""

    def on_edit(evt_id: int, eds: dict[str, Any], ):
        update_event(evt_id, eds)
        pass

    change = st.session_state['edited_todo_df']
    df = s_todo_df()
    if edited := change['edited_rows']:  # 编辑列属性 # {2: {'col_name': 'foo'}},
        for row_id, edits in edited.items():
            event_id = df.at[row_id, 'id']
            if 'tags' in edits.keys():
                st.toast("暂不支持编辑tags", icon='⚠️')
            on_edit(event_id, edits)
        pass
    if added := change['added_rows']:  # 新增列 [{'col_name':'bar'}]
        if added == [{}]:
            return  # 添加空行不保存直接返回不刷新 (避免打断输入)
        st.toast("暂不支持add记录", icon='⚠️')
        pass
    if deleted := change['deleted_rows']:  # 删除列 [1,2,3]
        for row_id in deleted:
            event_id = df.at[row_id, 'id']
            update_event(event_id, {'deleted': True})
        pass
    # 这里传入新的df, 将导致ui刷新
    reload_event_grid()  # 刷新 df


def fetch_event_df() -> pd.DataFrame:
    """从state中收集查询条件, 查询 todo_grid 所需的df"""
    return query_events(
        page=st.session_state['p_event_grid-page'],
        by=st.session_state['p_event_grid-by'],
        show_tag_system=st.session_state['p_event_grid-show_tag_system'],
    )


def fetch_todo_df() -> pd.DataFrame:
    """从state中收集查询条件, 查询event_grid 所需的df"""
    return query_todo_events(
        page=st.session_state['p_todo_grid-page'],
        by=st.session_state['p_todo_grid-by'],
        # show_tag_system=st.session_state['p_todo_grid-show_tag_system'],
    )


# === UI ===
def adp_data_editor_height(df_len: int, reserve=1, least_len=2) -> int:
    if df_len < least_len:
        df_len = least_len
    return 2 + (df_len + 1 + reserve) * 35


def ui_event_grid_query_param():
    with st.expander("参数", expanded=False):
        st.number_input('单页行数', value=20, min_value=10, max_value=100, step=10, key='p_event_grid-by',
                        on_change=reload_event_grid)
        st.toggle('`$SYSTEM` tag', key='p_event_grid-show_tag_system', help='展示包含$SYSTEM标签的事件',
                  on_change=reload_event_grid)


def ui_event_todo_query_param():
    with st.expander("参数", expanded=False):
        st.number_input('单页行数', value=20, min_value=10, max_value=100, step=10, key='p_todo_grid-by',
                        on_change=reload_event_grid)


@st.dialog("Create Event", width='large')
def ui_form_event(on_submitted: Callable[[Event, Iterable[Tag]], Any], now: datetime):
    with st.form("form_create_mind-event", clear_on_submit=True):
        content = st.text_area("Content")
        c1, c2 = st.columns([1, 1])
        with c1:
            record_at = st.date_input("Record Date", value=now)
        with c2:
            record_time = st.time_input("Record Time", value=now.time())
        record_at = datetime.combine(record_at, record_time)
        create_at = datetime.now()
        # tags
        selected_tags = st.multiselect("选择标签", options=Tag.select(), format_func=lambda x: x.name)
        if st.form_submit_button("Submit", type='primary', use_container_width=True):
            new_event = Event(
                content=content,
                record_at=record_at,
                create_at=create_at,
                update_at=datetime.now(),  # 由于填写表单耗时,必然大于create_at
            )
            if content.strip().__len__() != 0:  # 确保content非空
                st.toast(f"New Event {new_event.create_at}")
                on_submitted(new_event, selected_tags)
                reload_event_grid()
                st.rerun()
            else:
                st.warning("请填写内容")


@st.dialog("Batch Create Event", width='large')
def ui_form_event_batch(on_submitted: Callable[[Event, Iterable[Tag]], Any], now: datetime):
    """
    批量导入Event
    """
    data = st.text_area("Content", help='格式为 <record_time><split_char><content>\n')
    # ==
    record_at = st.date_input("Record Date", value=now, help='一次只能导入单日event')
    rst = []
    for ln in data.splitlines():
        if ln.strip().__len__() == 0:  # 跳过空白行
            continue
        logger.trace(f'#0#[{ln}]')

        rcd_at, ctt = ln[:5].rstrip(), ln[5:].lstrip()
        logger.trace(f'#1#[{rcd_at}][{ctt}]')

        if rcd_at.__contains__('点'): # 重新处理 ‘x点’ 时间
            sp = ln.split('点')
            rcd_at, ctt = sp[0], '点'.join(sp[0:])
            rcd_at = rcd_at[:-1] + ':00'
            ctt = f'大约) {ctt}'
            logger.trace(f'#2#{rcd_at},{ctt}')
        rst.append([rcd_at, ctt])
    rst = [{
        'record_at': datetime.combine(
            record_at,
            datetime.strptime(evt_data[0], '%H:%M').time()),
        'content': ''.join(evt_data[1:]),
    } for evt_data in rst]
    st.data_editor(
        pd.DataFrame(rst)
    )
    # tags
    # selected_tags = st.multiselect("选择标签", options=Tag.select(), format_func=lambda x: x.name)
    if st.button("Submit", type='primary', use_container_width=True):
        if rst.__len__() != 0:  # 确保content非空
            st.toast(f"Batch New Event {rst.__len__()}")
            n = datetime.now()
            for r in rst:
                e = Event.create(
                    **r,
                    create_at=n,
                    update_at=n,
                )
                on_submitted(e, [])
            reload_event_grid()
            st.rerun()
        else:
            st.warning("请填写内容")


@st.dialog("Create Tag", width='large')
def ui_form_tag(on_submitted: Callable[[Tag], None]):
    with st.form("form_create_tag", clear_on_submit=True):
        name = st.text_input("Name")
        desc = st.text_area("Desc")
        if st.form_submit_button("Submit Tag", type='primary', use_container_width=True):
            tag = Tag(name=name, desc=desc)
            if name.strip().__len__() != 0:
                st.toast(f"New Tag {tag.name}")
                on_submitted(tag)
                st.rerun()
            else:
                st.warning("请填写内容")


@st.dialog("Create Todo", width='large')
def ui_form_todo(on_submitted: Callable[[Event, TodoDTO], None]):
    content = st.text_area("Content")
    c0, c1, c2 = st.columns([1, 2, 2])
    with c0:
        st.toggle('截止日期', key='p_todo_due-at', label_visibility='collapsed')
    with c1:
        due_at = st.date_input("Due Date", disabled=not st.session_state['p_todo_due-at'])
    with c2:
        due_time = st.time_input("Due Time", disabled=not st.session_state['p_todo_due-at'])
    due_at = datetime.combine(due_at, due_time) if not st.session_state['p_todo_due-at'] else None
    todo = TodoDTO(due_at=due_at)
    # tags
    if st.button("Submit", type='primary', use_container_width=True):
        todo_event = Event(
            content=content,
            record_at=datetime.fromtimestamp(0),  # 待办的完成时间,0代表未完成
            create_at=datetime.now(),
            update_at=datetime.now(),
        )
        if content.strip().__len__() != 0:  # 确保content非空
            st.toast(f"New Todo {todo_event.content}")
            on_submitted(todo_event, todo)
            reload_todo_grid()
            st.rerun()
        else:
            st.warning("请填写内容")


def ui_event_grid():
    st.data_editor(
        s_event_df(), key='edited_event_df', use_container_width=True, height=adp_data_editor_height(len(s_event_df())),
        on_change=on_change_event_df, num_rows="dynamic",
        column_order=['content', 'tags', 'record_at', 'create_at', 'update_at', ],  # 'id',
        column_config={
            'content': st.column_config.TextColumn(width='large'),
            'tags': st.column_config.TextColumn(width='medium'),
            'record_at': st.column_config.DatetimeColumn(width='small', format='HH:mm YY-MM-DD'),
            'create_at': st.column_config.DatetimeColumn(width='small', disabled=True, format='YY-MM-DD HH:mm'),
            'update_at': st.column_config.DatetimeColumn(width='small', disabled=True, format='YY-MM-DD HH:mm'),
        },
    )
    pass


def ui_todo_grid():
    st.data_editor(
        s_todo_df(), key='edited_todo_df', use_container_width=True, height=adp_data_editor_height(len(s_event_df())),
        on_change=on_change_todo_df, num_rows="dynamic",
        column_order=['content', 'record_at', 'create_at', 'update_at', ],  # 'id', 'tags',
        column_config={
            'content': st.column_config.TextColumn(width='large'),
            # 'tags': st.column_config.TextColumn(width='medium'),
            'record_at': st.column_config.DatetimeColumn(width='small', label='Done at', format='HH:mm YY-MM-DD'),
            'create_at': st.column_config.DatetimeColumn(width='small', disabled=True, format='YY-MM-DD HH:mm'),
            'update_at': st.column_config.DatetimeColumn(width='small', disabled=True, format='YY-MM-DD HH:mm'),
        },
    )
    pass


def ui_tag_grid():
    df = query_tags()  #
    st.data_editor(
        df,
        column_order=['id', 'name', 'desc'],  # 'id',
        column_config={
            'id': st.column_config.NumberColumn(width='small'),
            'name': st.column_config.TextColumn(width='medium'),
            'desc': st.column_config.TextColumn(width='large'),
        },
        use_container_width=True,
        height=adp_data_editor_height(len(df)),
    )
    pass


# === 布局 ===

def main():
    st.set_page_config(layout='wide', )
    st.markdown(
        """
        # Event Base
        Everything is Event
        """)
    t1, t2, t3 = st.tabs(['Events', 'Todo', 'Tags'])

    # Tab === Events管理
    with t1:
        ui_event_grid_query_param()
        #
        t1c1, t1c2, t1c3, t1c4 = st.columns([1, 1, 1, 2])
        with t1c1:
            if st.button("Create Event"):
                ui_form_event(save_event, datetime.now())
        with t1c2:
            if st.button("Batch Create"):
                ui_form_event_batch(save_event, datetime.now())
            pass
        with t1c3:
            st.number_input("翻页", key='p_event_grid-page', label_visibility='collapsed', help='翻页',
                            value=1, step=1, min_value=1, on_change=reload_event_grid)
        with t1c4:
            st.container()
        # === === event表格
        ui_event_grid()

    # Tab === _Todo
    with t2:
        ui_event_todo_query_param()
        if st.button("Create Todo"):
            ui_form_todo(save_todo)
        st.number_input("翻页", key='p_todo_grid-page', label_visibility='collapsed', help='翻页',
                        value=1, step=1, min_value=1, on_change=reload_event_grid)
        # === === todo表格
        ui_todo_grid()
        pass
    # Tab === Tag 管理
    with t3:
        if st.button("Create Tag"):
            ui_form_tag(save_tag)
        # === === tag表格
        ui_tag_grid()
        pass


if __name__ == '__main__':
    import os
    import sys

    # logger init
    log_level = st.secrets['log']['level']
    logger.info(f"log_level#[{log_level}]")
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # database init
    if not os.path.exists(assets.db_uri_database()):
        from scripts.init import init_database

        init_database()

    # app run
    main()
