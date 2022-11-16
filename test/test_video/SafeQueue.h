/*
 * @Description: Productor and cumsumer queue.
 * @version: 1.0
 * @Author: Ricardo Lu<shenglu1202@163.com>
 * @Date: 2022-10-11 10:50:49
 * @LastEditors: Ricardo Lu
 * @LastEditTime: 2022-10-12 03:29:30
 */
#pragma once

#include <list>
#include <mutex>
#include <condition_variable>
#include <iostream>

template<typename T>
class SafetyQueue {
private:
    std::list<std::shared_ptr<T>> m_queue;
    std::mutex m_mutex;//全局互斥锁
    std::condition_variable_any m_notEmpty;//全局条件变量（不为空）
    std::condition_variable_any m_notFull;//全局条件变量（不为满）
    unsigned int m_maxSize;//队列最大容量

private:
    //队列为空
    bool isEmpty() const {
        return m_queue.empty();
    }
    //队列已满
    bool isFull() const {
        return m_queue.size() == m_maxSize;
    }

public:
    SafetyQueue(unsigned int maxSize = 25) {
        this->m_maxSize = maxSize;
    }

    int queuecount() const {
        return m_queue.size();
    }

    void setMaxSize(unsigned int maxSize) {
        this->m_maxSize = maxSize;
    }

    unsigned int getMaxSize() {
        return this->m_maxSize;
    }

    unsigned int getCurrentSize() {
        return m_queue.size();
    }

    ~SafetyQueue(){}

    void product(const std::shared_ptr<T>& v) {
        std::unique_lock<std::mutex> locker(m_mutex);
        while(isFull()) {
            m_notFull.wait(m_mutex);
        }

        m_queue.push_back(v);
        m_notEmpty.notify_one();
    }
    void consumption(std::shared_ptr<T>& v) {
        std::unique_lock<std::mutex> locker(m_mutex);
        while(isEmpty()) {
            m_notEmpty.wait(m_mutex);
        }

        v = m_queue.front();
        m_queue.pop_front();
        m_notFull.notify_one();
    }

    std::string debug_info;
};