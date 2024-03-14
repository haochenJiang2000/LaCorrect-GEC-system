<template>
    <div v-if="LoggedIn">
        <el-card class="box-card" style="width: 1000px;">
            <template #header>
            <div class="card-header">
                <span>纠错测试</span>
                <span>
                    <el-select v-model="GEC_type_test" placeholder="通用领域">
                        <el-option label="通用领域" value="通用领域" />
                        <el-option label="航空航天领域" value="航空航天领域" />
                    </el-select>
                    <el-select v-model="GEC_mode_test" placeholder="速度优先">
                        <el-option label="速度优先" value="速度优先" />
                        <el-option label="中度纠错（推荐）" value="中度纠错" />
                        <el-option label="深度纠错" value="深度纠错" disabled/>
                    </el-select>
                </span>
            </div>
            </template>
            <el-alert style="width: 958px;" title=
            "请在下方输入框输入待纠错的文本, 本页面仅供测试, 长文本请以文件形式提交."
            type="info" :closable="false"/>
            <el-alert style="width: 958px;" title=
            "用户每十分钟最多提交二十次文本，请勿连续提交重复内容。"
            type="info" :closable="false"/>
            <el-alert style="width: 958px;" title=
            "参考时长  速度优先：4-6s | 中度纠错：20-30s。 若使用人数过多等待时间会延长。由于资源和速度限制，深度纠错模式仅在文档级纠错中开放" 
            type="info" :closable="false"/>
            <span>&nbsp</span>
            <el-input
                v-model="test_text"
                :autosize="{ minRows: 3, maxRows: 5 }"
                maxlength="100"
                show-word-limit
                type="textarea"
                placeholder="请输入待纠错的文本, 本页面仅供测试, 长文本请以文件形式提交"
            />
            <div>
                &nbsp
            </div>
            <div v-if="submit">
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="primary" @click="Test_correct()" loading>提交</el-button>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="default" @click="test_text=''">重置</el-button>
            </div>
            <div v-else>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="primary" @click="Test_correct()">提交</el-button>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                <el-button type="default" @click="test_text=''">重置</el-button>
            </div>
        </el-card>
        <div>
            &nbsp
        </div>
        <div v-if="GetResult">
            <el-card class="box-card" style="width: 1000px;">
                <template #header>
                <div class="card-header">
                    <span style="font-size: 16px;">结果</span>
                </div>
                </template>
                <div class="text item">
                <div v-for="txt in test_result.tgt">
                    <div v-if="txt[1]==1" class="delete_text">{{ txt[0] }}</div>
                    <div v-else-if="txt[1]==2" class="blue_text">{{ txt[0] }}</div>
                    <div v-else class="normal_text">{{ txt[0] }}</div>
                </div>
                </div>
                <div>
                &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
                </div>
            </el-card>
        </div>
    </div>
    <div v-else>
        <el-space wrap>
            <el-card class="box-card" style="width: 1000px;">
                <template #header>
                <div class="card-header">
                    请先登录
                </div>
                </template>
                <div class="text item">
                <h1>纠错测试</h1>
                </div>
            </el-card>
        </el-space>  
    </div>
    
</template>

<script lang="ts" setup> 
    import axios from 'axios';
    import { computed } from 'vue'
    import { useStore } from 'vuex'
    import { ElMessage } from 'element-plus';
    import { ref } from 'vue'

    const store = useStore()
    const LoggedIn = computed(() => store.state.log.LoggedIn || false)

    // let TestResult = {
    //     tgt: []
    // }
    const test_text = ref("研究结果为空空导弹中制导律设计提供理论参考。")
    const test_result = computed(() => store.state.submit.TestResult)
    const GetResult = ref(false)
    const submit = ref(false)
    const GEC_mode_test = ref("中度纠错")
    const GEC_type_test = ref("航空航天领域")
    // let test_result = computed(() => store.state.result.test_result)

    // 后端请求对应的纠错结果
    const Test_correct = () => {
        submit.value = true
        let formData = new FormData();
        // console.log(test_text.value)
        // console.log(GEC_mode_test.value)
        formData.append("Sentences", test_text.value);
        formData.append("UserName", store.state.log.UserName);
        formData.append("GEC_mode_test", GEC_mode_test.value);
        formData.append("GEC_type_test", GEC_type_test.value);
        axios.post('/test', formData, {timeout: 60 * 60 * 1000}).then(res => {
            console.log("完成纠错")
            console.log(res.data)
            if(res.data.code != 707){
                console.log(res.data.error)
                if(res.data.error) {
                    store.commit('submit/UPDATE_TestResult', res.data.error)
                    console.log(test_result.value)
                    GetResult.value = true
                    submit.value = false
                }
            }
            else{
                store.commit('log/UPDATE_LOG', false)
                store.commit('log/UPDATE_EXIT', false)
            }
            if (res.data.message) {
            ElMessage(res.data.message)
            }
        })
    }
</script>

<style>

</style>